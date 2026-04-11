"""
Backtesting module — honest out-of-sample simulation on FDCO historical data.

To avoid in-sample bias the backtest uses a TEMPORAL HOLDOUT: matches are
sorted by date and only the most recent 30% are evaluated (the model was
trained predominantly on older data). This is still partially in-sample but
gives a far more realistic picture than testing on all data.

Strategies simulated:
  flat   — £1 on the predicted outcome every match
  value  — £1 only when model probability exceeds implied prob by ≥ min_edge
  kelly  — Fractional Kelly on value bets (hard-capped to prevent blow-up)

Breakdowns:
  by_league     — ROI per competition (PL, PD, BL1, …)
  by_confidence — performance per star-rating tier (1★ to 5★)
  monthly_pnl   — flat + value P&L month-by-month
  side_markets  — Over/Under 2.5 flat staking (when B365 O/U odds available)
"""
import os
import numpy as np
import joblib
from typing import Optional

ML_DIR = os.path.dirname(__file__)

RESULT_MODEL_PATH = os.path.join(ML_DIR, "result_model.joblib")
RESULT_CAL_PATH   = os.path.join(ML_DIR, "result_calibrator.joblib")
GOALS_MODEL_PATH  = os.path.join(ML_DIR, "goals_model.joblib")
GOALS_CAL_PATH    = os.path.join(ML_DIR, "goals_calibrator.joblib")
BTTS_MODEL_PATH   = os.path.join(ML_DIR, "btts_model.joblib")
BTTS_CAL_PATH     = os.path.join(ML_DIR, "btts_calibrator.joblib")

# Confidence tiers: (label, min_prob, max_prob)
CONFIDENCE_TIERS = [
    ("5★ (≥70%)",  0.70, 1.01),
    ("4★ (60-70%)", 0.60, 0.70),
    ("3★ (50-60%)", 0.50, 0.60),
    ("low (<50%)",  0.00, 0.50),
]


def _load_model(cal_path: str, raw_path: str):
    if os.path.exists(cal_path):
        return joblib.load(cal_path)
    if os.path.exists(raw_path):
        return joblib.load(raw_path)
    return None


def _empty_strat() -> dict:
    return {"bets": 0, "wins": 0, "pnl": 0.0, "peak": 0.0, "max_dd": 0.0}


def _update_dd(s: dict, cumulative_pnl: float) -> None:
    """Update peak and max drawdown for a strategy tracking dict."""
    if cumulative_pnl > s["peak"]:
        s["peak"] = cumulative_pnl
    dd = s["peak"] - cumulative_pnl
    if dd > s["max_dd"]:
        s["max_dd"] = dd


def _roi(pnl: float, n: int) -> float:
    return round(pnl / n * 100, 1) if n else 0.0


def _wr(wins: int, n: int) -> float:
    return round(wins / n * 100, 1) if n else 0.0


def _strat_summary(s: dict) -> dict:
    return {
        "bets":          s["bets"],
        "wins":          s["wins"],
        "win_rate":      _wr(s["wins"], s["bets"]),
        "pnl":           round(s["pnl"], 2),
        "roi":           _roi(s["pnl"], s["bets"]),
        "max_drawdown":  round(s["max_dd"], 2),
    }


def run_backtest(
    holdout_fraction: float = 0.30,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
    starting_bankroll: float = 100.0,
    min_odds: float = 1.20,
    max_odds: float = 12.0,
) -> dict:
    """
    Simulate staking on FDCO historical data (temporal holdout).
    Returns a summary dict with ROI, P&L, drawdown, and breakdowns.
    """
    from ml.fdco_trainer import build_fdco_training_data

    result_model = _load_model(RESULT_CAL_PATH, RESULT_MODEL_PATH)
    goals_model  = _load_model(GOALS_CAL_PATH,  GOALS_MODEL_PATH)

    if result_model is None:
        return {"error": "No trained model found. Run ml/train.py first."}

    X, y_result, y_goals, _, odds_rows = build_fdco_training_data(min_history=5)

    if len(X) == 0:
        return {"error": "No FDCO data found. Run ml/train.py first to download CSVs."}

    # ── Temporal holdout: sort by date, keep last 30% for testing ────────────
    dated = [(r.get("date", ""), i) for i, r in enumerate(odds_rows)]
    dated.sort(key=lambda x: x[0])
    cutoff = int(len(dated) * (1 - holdout_fraction))
    holdout_indices = {orig_i for _, orig_i in dated[cutoff:]}

    test_rows = [r for i, r in enumerate(odds_rows) if i in holdout_indices]

    if not test_rows:
        return {"error": "No holdout data available."}

    # ── Batch predict ─────────────────────────────────────────────────────────
    test_X_indices = [r["feature_idx"] for r in test_rows]
    test_X = X[test_X_indices]

    result_probs = result_model.predict_proba(test_X)   # (N, 3)
    goals_probs  = goals_model.predict_proba(test_X) if goals_model else None  # (N, 2)

    idx_to_outcome = {0: "HOME", 1: "DRAW", 2: "AWAY"}

    # ── Strategy accumulators ─────────────────────────────────────────────────
    flat  = _empty_strat()
    value = _empty_strat()
    kelly = _empty_strat()
    bankroll = starting_bankroll
    peak_bankroll = starting_bankroll

    by_league:     dict[str, dict] = {}   # league → {flat: strat, value: strat}
    by_confidence: dict[str, dict] = {    # tier_label → {flat: strat, value: strat}
        label: {"flat": _empty_strat(), "value": _empty_strat()}
        for label, _, _ in CONFIDENCE_TIERS
    }
    monthly_flat:  dict[str, dict] = {}   # "YYYY-MM" → strat
    monthly_value: dict[str, dict] = {}

    # ── Over/Under side market ────────────────────────────────────────────────
    ou25 = _empty_strat()

    for local_i, row in enumerate(test_rows):
        p = result_probs[local_i]
        pred_idx  = int(np.argmax(p))
        pred_prob = float(p[pred_idx])
        won       = (pred_idx == int(row["result_label"]))

        league  = row.get("league", "?")
        month   = row.get("date", "????-??")[:7]

        # Confidence tier label
        tier_label = "low (<50%)"
        for lbl, lo, hi in CONFIDENCE_TIERS:
            if lo <= pred_prob < hi:
                tier_label = lbl
                break

        pred_outcome = idx_to_outcome[pred_idx]
        odds_key = {"HOME": "b365h", "DRAW": "b365d", "AWAY": "b365a"}[pred_outcome]
        decimal_odds = row.get(odds_key)

        if not decimal_odds or decimal_odds < min_odds or decimal_odds > max_odds:
            pass  # still try O/U below even if result odds are bad
        else:
            implied_prob = 1.0 / decimal_odds
            edge = pred_prob - implied_prob

            # Per-league init
            if league not in by_league:
                by_league[league] = {
                    "flat": _empty_strat(), "value": _empty_strat()
                }
            # Per-month init
            if month not in monthly_flat:
                monthly_flat[month]  = _empty_strat()
                monthly_value[month] = _empty_strat()

            # ── Flat staking ─────────────────────────────────────────────────
            flat["bets"] += 1
            by_league[league]["flat"]["bets"] += 1
            by_confidence[tier_label]["flat"]["bets"] += 1
            monthly_flat[month]["bets"] += 1

            if won:
                profit = decimal_odds - 1.0
                flat["pnl"]  += profit; flat["wins"]  += 1
                by_league[league]["flat"]["pnl"]  += profit
                by_league[league]["flat"]["wins"] += 1
                by_confidence[tier_label]["flat"]["pnl"]  += profit
                by_confidence[tier_label]["flat"]["wins"] += 1
                monthly_flat[month]["pnl"]  += profit
                monthly_flat[month]["wins"] += 1
            else:
                flat["pnl"] -= 1.0
                by_league[league]["flat"]["pnl"]  -= 1.0
                by_confidence[tier_label]["flat"]["pnl"] -= 1.0
                monthly_flat[month]["pnl"] -= 1.0

            _update_dd(flat, flat["pnl"])

            # ── Value staking ─────────────────────────────────────────────────
            if edge >= min_edge:
                value["bets"] += 1
                by_league[league]["value"]["bets"] += 1
                by_confidence[tier_label]["value"]["bets"] += 1
                monthly_value[month]["bets"] += 1

                if won:
                    profit = decimal_odds - 1.0
                    value["pnl"]  += profit; value["wins"]  += 1
                    by_league[league]["value"]["pnl"]  += profit
                    by_league[league]["value"]["wins"] += 1
                    by_confidence[tier_label]["value"]["pnl"]  += profit
                    by_confidence[tier_label]["value"]["wins"] += 1
                    monthly_value[month]["pnl"]  += profit
                    monthly_value[month]["wins"] += 1
                else:
                    value["pnl"] -= 1.0
                    by_league[league]["value"]["pnl"]  -= 1.0
                    by_confidence[tier_label]["value"]["pnl"] -= 1.0
                    monthly_value[month]["pnl"] -= 1.0

                _update_dd(value, value["pnl"])

                # ── Kelly staking ─────────────────────────────────────────────
                denom = 1.0 - (1.0 / decimal_odds)
                if denom > 0 and bankroll > 0:
                    kelly_pct   = kelly_fraction * edge / denom
                    kelly_pct   = min(kelly_pct, 0.05)
                    kelly_stake = min(bankroll * kelly_pct, bankroll)

                    if kelly_stake > 0:
                        kelly["bets"] += 1
                        if won:
                            gain = kelly_stake * (decimal_odds - 1.0)
                            kelly["pnl"] += gain; kelly["wins"] += 1
                            bankroll += gain
                        else:
                            kelly["pnl"] -= kelly_stake
                            bankroll     -= kelly_stake

                        if bankroll > peak_bankroll:
                            peak_bankroll = bankroll
                        dd_pct = (peak_bankroll - bankroll) / peak_bankroll * 100
                        kelly["max_dd"] = max(kelly["max_dd"], dd_pct)

        # ── Over/Under 2.5 (flat, when odds available) ────────────────────────
        if goals_probs is not None:
            ou_pred   = goals_probs[local_i][1] >= 0.5   # True = predict Over
            ou_actual = row["goals_label"] == 1           # True = actual Over
            ou_odds_key = "b365_o25" if ou_pred else "b365_u25"
            ou_dec = row.get(ou_odds_key)

            if ou_dec and min_odds <= ou_dec <= max_odds:
                ou25["bets"] += 1
                if ou_pred == ou_actual:
                    ou25["pnl"]  += ou_dec - 1.0
                    ou25["wins"] += 1
                else:
                    ou25["pnl"] -= 1.0
                _update_dd(ou25, ou25["pnl"])

    # ── Assemble result ───────────────────────────────────────────────────────
    by_league_out = {
        lg: {
            "flat":  _strat_summary(stats["flat"]),
            "value": _strat_summary(stats["value"]),
        }
        for lg, stats in sorted(by_league.items())
    }

    by_confidence_out = {
        label: {
            "flat":  _strat_summary(by_confidence[label]["flat"]),
            "value": _strat_summary(by_confidence[label]["value"]),
        }
        for label, _, _ in CONFIDENCE_TIERS
        if by_confidence[label]["flat"]["bets"] > 0
    }

    monthly_out = sorted(
        [
            {
                "month":      m,
                "flat_pnl":   round(monthly_flat[m]["pnl"], 2),
                "flat_bets":  monthly_flat[m]["bets"],
                "value_pnl":  round(monthly_value[m]["pnl"], 2),
                "value_bets": monthly_value[m]["bets"],
            }
            for m in monthly_flat
        ],
        key=lambda x: x["month"],
    )

    return {
        "total_matches": len(test_rows),
        "holdout_pct":   int(holdout_fraction * 100),
        "flat": {
            **_strat_summary(flat),
            "min_odds": min_odds,
            "max_odds": max_odds,
        },
        "value": {
            **_strat_summary(value),
            "min_edge_pct": round(min_edge * 100, 1),
        },
        "kelly": {
            "bets":              kelly["bets"],
            "wins":              kelly["wins"],
            "win_rate":          _wr(kelly["wins"], kelly["bets"]),
            "pnl":               round(kelly["pnl"], 2),
            "final_bankroll":    round(bankroll, 2),
            "starting_bankroll": starting_bankroll,
            "max_drawdown_pct":  round(min(kelly["max_dd"], 100.0), 1),
            "fraction":          kelly_fraction,
        },
        "by_league":     by_league_out,
        "by_confidence": by_confidence_out,
        "monthly_pnl":   monthly_out,
        "side_markets": {
            "over_under_25": {
                **_strat_summary(ou25),
                "available": goals_model is not None,
            }
        },
    }


if __name__ == "__main__":
    import json
    print("Running backtest (temporal holdout — most recent 30% of data)…")
    result = run_backtest()
    print(json.dumps(result, indent=2))
