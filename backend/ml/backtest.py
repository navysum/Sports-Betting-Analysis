"""
Backtesting module — honest out-of-sample simulation on FDCO historical data.

To avoid in-sample bias the backtest uses a TEMPORAL HOLDOUT: matches are
sorted by date and only the most recent 30% are evaluated (the model was
trained predominantly on older data). This is still partially in-sample but
gives a far more realistic picture than testing on all data.

FIX #10 — Full production pipeline:
Previously the backtest used raw XGBoost result probabilities only. This
was misleading because the live system blends DC + XGBoost and uses the
calibrated models. The backtest now mirrors the production stack:
  • All four calibrated XGBoost models (result, over25, btts, over35)
  • Dixon-Coles model for each market
  • Blend weights from blend_weights.json
  • Devigged implied probabilities (same as predict.py)
  • Multi-market staking evaluation

FIX #23 — Binomial confidence intervals:
All win rates now carry 95% Wilson score confidence intervals. Wilson CIs
are the standard frequentist choice for small-sample proportions (superior
to Wald/normal approximation at win rates near 0 or 1).

FIX #3 — Kelly formula alignment:
Backtest now uses the same standard Kelly formula as predict.py:
  kelly = fraction × (b×p − q) / b   where b=odds−1, q=1−p
Previously used fraction × edge / (1 − 1/odds), which differs when the
market has a margin (fair_p ≠ 1/odds after devigging).

FIX #4 — Over/Under 2.5 staking:
O/U 2.5 market is now evaluated using B365>2.5 / B365<2.5 odds from FDCO
CSVs. Reports flat and value ROI for over25, giving a more complete picture
of the ensemble's edge on goals markets.
  min_edge default raised 3% → 5% — stricter filter reduces false positives.

Strategies simulated:
  flat   — £1 on the predicted outcome every match
  value  — £1 only when model probability exceeds fair implied by ≥ min_edge
  kelly  — Fractional Kelly on value bets (hard-capped to prevent blow-up)

Run from backend/ directory:
    python -m ml.backtest
"""
import json
import os
import math
import numpy as np
import joblib
from typing import Optional

ML_DIR   = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(ML_DIR), "data")

# ─── Model loaders ────────────────────────────────────────────────────────────

def _load_model(name: str):
    """Load calibrator if available, else raw model, else None."""
    cal_path   = os.path.join(ML_DIR, f"{name}_calibrator.joblib")
    model_path = os.path.join(ML_DIR, f"{name}_model.joblib")
    if os.path.exists(cal_path):
        return joblib.load(cal_path)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def _load_blend_weights() -> dict:
    path = os.path.join(DATA_DIR, "blend_weights.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Sensible defaults if blend_weights.json hasn't been generated yet
    return {"result": 0.3, "over25": 0.3, "btts": 0.2, "over35": 0.25}


# ─── Devigging helpers ────────────────────────────────────────────────────────

def _devig_1x2(h_odds: float, d_odds: float, a_odds: float):
    """Remove bookmaker margin from 1X2 odds; return fair (home, draw, away)."""
    total = (1.0 / h_odds) + (1.0 / d_odds) + (1.0 / a_odds)
    return (1.0 / h_odds) / total, (1.0 / d_odds) / total, (1.0 / a_odds) / total


def _devig_binary(odds_a: float, odds_b: Optional[float] = None) -> float:
    """
    Remove margin from binary market.
    If both sides' odds are known use the exact two-outcome formula;
    otherwise approximate with a 2.5% margin.
    """
    if odds_b is not None and odds_b > 1.0:
        total = (1.0 / odds_a) + (1.0 / odds_b)
        return (1.0 / odds_a) / total
    return min(max((1.0 / odds_a) / 1.025, 0.01), 0.99)


# ─── Wilson score confidence interval ────────────────────────────────────────

def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    95% Wilson score confidence interval for a proportion.

    Returns (lower, upper) as percentages rounded to 1dp.
    Superior to Wald (normal approximation) when n is small or win_rate
    is near 0 or 1 — which is common for draw predictions.
    """
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, centre - margin) * 100
    hi = min(1.0, centre + margin) * 100
    return round(lo, 1), round(hi, 1)


# ─── Main backtest ────────────────────────────────────────────────────────────

def run_backtest(
    holdout_fraction: float = 0.30,   # test on most recent 30% of data
    min_edge: float = 0.05,           # minimum probability edge for value bets (raised 3%→5%)
    kelly_fraction: float = 0.25,     # quarter Kelly
    starting_bankroll: float = 100.0,
    min_odds: float = 1.20,
    max_odds: float = 12.0,           # filter out data errors / extreme outliers
) -> dict:
    """
    Simulate staking on FDCO historical data (temporal holdout).

    Returns a summary dict with ROI, P&L, drawdown, win rates per strategy.
    """
    from ml.fdco_trainer import build_fdco_training_data
    from ml.dixon_coles import load_dc_model

    # ── Load models ───────────────────────────────────────────────────────────
    result_cal  = _load_model("result")
    goals_cal   = _load_model("goals")
    btts_cal    = _load_model("btts")
    over35_cal  = _load_model("over35")

    if any(m is None for m in [result_cal, goals_cal, btts_cal, over35_cal]):
        return {"error": "One or more models missing. Run ml/train.py first."}

    dc_model      = load_dc_model()
    blend_weights = _load_blend_weights()

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y_result, y_goals, y_btts, y_over35, odds_rows, _ = build_fdco_training_data(
        min_history=5
    )

    if len(X) == 0:
        return {"error": "No FDCO data found. Run ml/train.py first to download CSVs."}

    # ── Temporal holdout: sort by date, keep last holdout_fraction for testing ─
    dated = [(r.get("date", ""), i) for i, r in enumerate(odds_rows)]
    dated.sort(key=lambda x: x[0])
    cutoff = int(len(dated) * (1 - holdout_fraction))
    holdout_indices = {orig_i for _, orig_i in dated[cutoff:]}

    test_rows = [r for i, r in enumerate(odds_rows) if i in holdout_indices]

    if not test_rows:
        return {"error": "No holdout data available."}

    # ── Batch XGBoost probabilities ───────────────────────────────────────────
    test_X_idx = [r["feature_idx"] for r in test_rows]
    test_X     = X[test_X_idx]

    xgb_result  = result_cal.predict_proba(test_X)        # (N, 3)
    xgb_over25  = goals_cal.predict_proba(test_X)[:, 1]   # (N,)
    xgb_btts    = btts_cal.predict_proba(test_X)[:, 1]
    xgb_over35  = over35_cal.predict_proba(test_X)[:, 1]

    # ── Blend weights ─────────────────────────────────────────────────────────
    w_res   = blend_weights.get("result", 0.3)
    w_ou25  = blend_weights.get("over25", 0.3)
    w_btts  = blend_weights.get("btts",   0.2)
    w_ou35  = blend_weights.get("over35", 0.25)

    # ── Iterate holdout matches ───────────────────────────────────────────────
    idx_to_outcome = {0: "HOME", 1: "DRAW", 2: "AWAY"}

    flat_pnl  = 0.0;  value_pnl = 0.0;  kelly_pnl = 0.0
    bankroll  = starting_bankroll

    flat_bets  = flat_wins  = 0
    value_bets = value_wins = 0
    kelly_bets = kelly_wins = 0

    peak_bankroll = starting_bankroll
    max_drawdown  = 0.0

    # Over/Under 2.5 market staking accumulators (FIX #4: added O/U staking)
    ou_flat_pnl  = 0.0;  ou_value_pnl = 0.0
    ou_flat_bets = ou_flat_wins = 0
    ou_val_bets  = ou_val_wins  = 0

    # Multi-market Brier accumulators
    brier_result = []; brier_over25 = []; brier_btts = []; brier_over35 = []
    n_dc_fallback = 0

    for local_i, row in enumerate(test_rows):
        b365h = row.get("b365h")
        b365d = row.get("b365d")
        b365a = row.get("b365a")

        # ── DC probabilities for this match ──────────────────────────────────
        dc_info = None
        if dc_model is not None:
            dc_info = dc_model.match_probs(
                row["home"], row["away"], league=row.get("league", "")
            )
        if dc_info:
            dc_h    = dc_info["home"]
            dc_d    = dc_info["draw"]
            dc_a    = dc_info["away"]
            dc_ou25 = dc_info.get("over25", 0.5)
            dc_bt   = dc_info.get("btts",   0.5)
            dc_ou35 = dc_info.get("over35", 0.35)
        else:
            dc_h = dc_d = dc_a = 1 / 3
            dc_ou25 = 0.5; dc_bt = 0.5; dc_ou35 = 0.35
            n_dc_fallback += 1

        # ── Blend ─────────────────────────────────────────────────────────────
        xr = xgb_result[local_i]
        blended_res   = (1 - w_res) * xr + w_res * np.array([dc_h, dc_d, dc_a])
        # Renormalise result probs
        total = blended_res.sum()
        if total > 0:
            blended_res /= total

        blended_ou25  = (1 - w_ou25) * xgb_over25[local_i]  + w_ou25  * dc_ou25
        blended_btts  = (1 - w_btts) * xgb_btts[local_i]    + w_btts  * dc_bt
        blended_ou35  = (1 - w_ou35) * xgb_over35[local_i]  + w_ou35  * dc_ou35

        # ── Multi-market Brier scores ─────────────────────────────────────────
        rl = int(row["result_label"])
        one_hot = np.zeros(3); one_hot[rl] = 1.0
        brier_result.append(float(np.sum((blended_res - one_hot) ** 2)))
        brier_over25.append((blended_ou25 - int(row["goals_label"])) ** 2)
        brier_btts.append(  (blended_btts  - int(row["btts_label"])) ** 2)
        brier_over35.append((blended_ou35  - int(row.get("over35_label", 0))) ** 2)

        # ── Over/Under 2.5 staking (FIX #4: independent of 1X2 odds availability) ──
        # Evaluated first so `continue` in the result block below doesn't skip it.
        ou_over_odds  = row.get("b365_over25")
        ou_under_odds = row.get("b365_under25")
        if (ou_over_odds and ou_under_odds
                and min_odds <= ou_over_odds <= max_odds
                and min_odds <= ou_under_odds <= max_odds):
            # Pick the side the model favours
            if blended_ou25 >= 0.5:
                ou_pred_prob  = blended_ou25
                ou_book_odds  = ou_over_odds
                ou_other_odds = ou_under_odds
                ou_won        = (row["goals_label"] == 1)  # over25 actual
            else:
                ou_pred_prob  = 1.0 - blended_ou25
                ou_book_odds  = ou_under_odds
                ou_other_odds = ou_over_odds
                ou_won        = (row["goals_label"] == 0)  # under25 actual

            ou_fair = _devig_binary(ou_book_odds, ou_other_odds)
            ou_edge = ou_pred_prob - ou_fair if ou_fair else 0.0

            # Flat: every match
            ou_flat_bets += 1
            if ou_won:
                ou_flat_pnl += ou_book_odds - 1.0;  ou_flat_wins += 1
            else:
                ou_flat_pnl -= 1.0

            # Value: only when edge ≥ min_edge
            if ou_edge >= min_edge:
                ou_val_bets += 1
                if ou_won:
                    ou_value_pnl += ou_book_odds - 1.0;  ou_val_wins += 1
                else:
                    ou_value_pnl -= 1.0

        # ── Result market staking (B365 1X2) ─────────────────────────────────
        if not all([b365h, b365d, b365a]):
            continue
        if not (min_odds <= b365h <= max_odds and
                min_odds <= b365d <= max_odds and
                min_odds <= b365a <= max_odds):
            continue

        pred_idx  = int(np.argmax(blended_res))
        pred_prob = float(blended_res[pred_idx])
        won       = (pred_idx == rl)
        pred_outcome = idx_to_outcome[pred_idx]
        odds_key  = {"HOME": "b365h", "DRAW": "b365d", "AWAY": "b365a"}[pred_outcome]
        decimal_odds = row.get(odds_key)

        if not decimal_odds or decimal_odds < min_odds or decimal_odds > max_odds:
            continue

        # Devigged fair implied probability for this outcome
        fair_h, fair_d, fair_a = _devig_1x2(b365h, b365d, b365a)
        fair_implied = {"HOME": fair_h, "DRAW": fair_d, "AWAY": fair_a}[pred_outcome]
        edge = pred_prob - fair_implied

        # ── Flat staking ──────────────────────────────────────────────────────
        flat_bets += 1
        if won:
            flat_pnl += decimal_odds - 1.0; flat_wins += 1
        else:
            flat_pnl -= 1.0

        # ── Value staking ─────────────────────────────────────────────────────
        if edge >= min_edge:
            value_bets += 1
            if won:
                value_pnl += decimal_odds - 1.0; value_wins += 1
            else:
                value_pnl -= 1.0

            # ── Fractional Kelly staking (FIX #3: standard Kelly matching predict.py) ──
            # predict.py uses: kelly = (b*p - q) / b  where b=odds-1, q=1-p
            # Old backtest used: kelly = fraction * edge / (1 - 1/odds) — differs with margin
            b = decimal_odds - 1.0
            if b > 0:
                kelly = kelly_fraction * ((b * pred_prob - (1.0 - pred_prob)) / b)
                kelly_pct   = max(0.0, min(kelly, 0.05))   # floor 0, cap 5% of bankroll
                kelly_stake = min(bankroll * kelly_pct, bankroll)

                if kelly_stake > 0 and bankroll > 0:
                    kelly_bets += 1
                    if won:
                        kelly_pnl += kelly_stake * (decimal_odds - 1.0)
                        bankroll  += kelly_stake * (decimal_odds - 1.0)
                        kelly_wins += 1
                    else:
                        kelly_pnl -= kelly_stake
                        bankroll  -= kelly_stake

                    if bankroll > peak_bankroll:
                        peak_bankroll = bankroll
                    if peak_bankroll > 0:
                        drawdown     = (peak_bankroll - bankroll) / peak_bankroll * 100
                        max_drawdown = max(max_drawdown, drawdown)

    # ── Summary helpers ───────────────────────────────────────────────────────
    def _roi(pnl, n):
        return round(pnl / n * 100, 1) if n else 0.0

    def _wr(wins, n):
        return round(wins / n * 100, 1) if n else 0.0

    n_test = len(test_rows)
    return {
        "total_matches":   n_test,
        "holdout_pct":     int(holdout_fraction * 100),
        "dc_fallback_pct": round(n_dc_fallback / max(n_test, 1) * 100, 1),
        # ── Multi-market calibration ──────────────────────────────────────────
        "brier": {
            "result": round(float(np.mean(brier_result)), 4) if brier_result else None,
            "over25": round(float(np.mean(brier_over25)), 4) if brier_over25 else None,
            "btts":   round(float(np.mean(brier_btts)),   4) if brier_btts   else None,
            "over35": round(float(np.mean(brier_over35)), 4) if brier_over35 else None,
        },
        # ── Result market staking results ─────────────────────────────────────
        "flat": {
            "bets":          flat_bets,
            "wins":          flat_wins,
            "win_rate":      _wr(flat_wins, flat_bets),
            # FIX #23: Wilson score 95% CI on win rate
            "win_rate_ci95": _wilson_ci(flat_wins, flat_bets),
            "pnl":           round(flat_pnl, 2),
            "roi":           _roi(flat_pnl, flat_bets),
        },
        "value": {
            "bets":          value_bets,
            "wins":          value_wins,
            "win_rate":      _wr(value_wins, value_bets),
            "win_rate_ci95": _wilson_ci(value_wins, value_bets),
            "pnl":           round(value_pnl, 2),
            "roi":           _roi(value_pnl, value_bets),
            "min_edge_pct":  round(min_edge * 100, 1),
        },
        "kelly": {
            "bets":              kelly_bets,
            "wins":              kelly_wins,
            "win_rate":          _wr(kelly_wins, kelly_bets),
            "win_rate_ci95":     _wilson_ci(kelly_wins, kelly_bets),
            "pnl":               round(kelly_pnl, 2),
            "final_bankroll":    round(bankroll, 2),
            "starting_bankroll": starting_bankroll,
            "max_drawdown_pct":  round(min(max_drawdown, 100.0), 1),
            "fraction":          kelly_fraction,
        },
        # ── Over/Under 2.5 staking results (FIX #4) ──────────────────────────
        "over25_flat": {
            "bets":     ou_flat_bets,
            "wins":     ou_flat_wins,
            "win_rate": _wr(ou_flat_wins, ou_flat_bets),
            "win_rate_ci95": _wilson_ci(ou_flat_wins, ou_flat_bets),
            "pnl":      round(ou_flat_pnl, 2),
            "roi":      _roi(ou_flat_pnl, ou_flat_bets),
        },
        "over25_value": {
            "bets":         ou_val_bets,
            "wins":         ou_val_wins,
            "win_rate":     _wr(ou_val_wins, ou_val_bets),
            "win_rate_ci95": _wilson_ci(ou_val_wins, ou_val_bets),
            "pnl":          round(ou_value_pnl, 2),
            "roi":          _roi(ou_value_pnl, ou_val_bets),
            "min_edge_pct": round(min_edge * 100, 1),
        },
    }


if __name__ == "__main__":
    print("Running backtest (temporal holdout — most recent 30% of data)…")
    result = run_backtest()
    print(json.dumps(result, indent=2))
