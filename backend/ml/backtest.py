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

Run from backend/ directory:
    python -m ml.backtest
"""
import os
import numpy as np
import joblib
from typing import Optional

ML_DIR = os.path.dirname(__file__)
RESULT_MODEL_PATH = os.path.join(ML_DIR, "result_model.joblib")
RESULT_CAL_PATH   = os.path.join(ML_DIR, "result_calibrator.joblib")


def _load_predictor():
    if os.path.exists(RESULT_CAL_PATH):
        return joblib.load(RESULT_CAL_PATH)
    if os.path.exists(RESULT_MODEL_PATH):
        return joblib.load(RESULT_MODEL_PATH)
    return None


def run_backtest(
    holdout_fraction: float = 0.30,   # test on most recent 30% of data
    min_edge: float = 0.03,           # minimum probability edge for value bets
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

    model = _load_predictor()
    if model is None:
        return {"error": "No trained model found. Run ml/train.py first."}

    X, y_result, _, _, odds_rows = build_fdco_training_data(min_history=5)

    if len(X) == 0:
        return {"error": "No FDCO data found. Run ml/train.py first to download CSVs."}

    # ── Temporal holdout: sort by date, keep last 30% for testing ────────────
    dated = [(r.get("date", ""), i) for i, r in enumerate(odds_rows)]
    dated.sort(key=lambda x: x[0])
    cutoff = int(len(dated) * (1 - holdout_fraction))
    holdout_indices = {orig_i for _, orig_i in dated[cutoff:]}

    test_rows = [
        r for i, r in enumerate(odds_rows)
        if i in holdout_indices
    ]

    if not test_rows:
        return {"error": "No holdout data available."}

    # Predict all at once
    test_X_indices = [r["feature_idx"] for r in test_rows]
    test_X = X[test_X_indices]
    probs = model.predict_proba(test_X)  # (N, 3) → [home_p, draw_p, away_p]

    idx_to_outcome = {0: "HOME", 1: "DRAW", 2: "AWAY"}

    flat_pnl  = 0.0
    value_pnl = 0.0
    kelly_pnl = 0.0
    bankroll  = starting_bankroll

    flat_bets  = flat_wins  = 0
    value_bets = value_wins = 0
    kelly_bets = kelly_wins = 0

    peak_bankroll = starting_bankroll
    max_drawdown  = 0.0

    for local_i, row in enumerate(test_rows):
        p = probs[local_i]
        pred_idx  = int(np.argmax(p))
        pred_prob = float(p[pred_idx])
        won       = (pred_idx == int(row["result_label"]))

        pred_outcome = idx_to_outcome[pred_idx]
        odds_key = {"HOME": "b365h", "DRAW": "b365d", "AWAY": "b365a"}[pred_outcome]
        decimal_odds = row.get(odds_key)

        # Skip missing or implausible odds
        if not decimal_odds or decimal_odds < min_odds or decimal_odds > max_odds:
            continue

        # ── Flat staking ─────────────────────────────────────────────────────
        flat_bets += 1
        if won:
            flat_pnl += decimal_odds - 1.0
            flat_wins += 1
        else:
            flat_pnl -= 1.0

        # ── Value staking ─────────────────────────────────────────────────────
        implied_prob = 1.0 / decimal_odds
        edge = pred_prob - implied_prob

        if edge >= min_edge:
            value_bets += 1
            if won:
                value_pnl += decimal_odds - 1.0
                value_wins += 1
            else:
                value_pnl -= 1.0

            # ── Kelly staking ─────────────────────────────────────────────────
            # Kelly % = fraction × edge / (1 - 1/odds)
            # Hard caps to prevent compound blow-up
            denominator = 1.0 - (1.0 / decimal_odds)
            if denominator <= 0:
                continue

            kelly_pct = kelly_fraction * edge / denominator
            kelly_pct = min(kelly_pct, 0.05)           # max 5% of bankroll per bet
            kelly_stake = bankroll * kelly_pct
            kelly_stake = min(kelly_stake, bankroll)    # can't bet more than you have

            if kelly_stake <= 0 or bankroll <= 0:
                continue

            kelly_bets += 1
            if won:
                kelly_pnl += kelly_stake * (decimal_odds - 1.0)
                bankroll  += kelly_stake * (decimal_odds - 1.0)
                kelly_wins += 1
            else:
                kelly_pnl -= kelly_stake
                bankroll  -= kelly_stake

            # Track drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            if peak_bankroll > 0:
                drawdown = (peak_bankroll - bankroll) / peak_bankroll * 100
                max_drawdown = max(max_drawdown, drawdown)

    def _roi(pnl, n):
        return round(pnl / n * 100, 1) if n else 0.0

    def _wr(wins, n):
        return round(wins / n * 100, 1) if n else 0.0

    return {
        "total_matches": len(test_rows),
        "holdout_pct": int(holdout_fraction * 100),
        "flat": {
            "bets":     flat_bets,
            "wins":     flat_wins,
            "win_rate": _wr(flat_wins, flat_bets),
            "pnl":      round(flat_pnl, 2),
            "roi":      _roi(flat_pnl, flat_bets),
        },
        "value": {
            "bets":         value_bets,
            "wins":         value_wins,
            "win_rate":     _wr(value_wins, value_bets),
            "pnl":          round(value_pnl, 2),
            "roi":          _roi(value_pnl, value_bets),
            "min_edge_pct": round(min_edge * 100, 1),
        },
        "kelly": {
            "bets":              kelly_bets,
            "wins":              kelly_wins,
            "win_rate":          _wr(kelly_wins, kelly_bets),
            "pnl":               round(kelly_pnl, 2),
            "final_bankroll":    round(bankroll, 2),
            "starting_bankroll": starting_bankroll,
            "max_drawdown_pct":  round(min(max_drawdown, 100.0), 1),
            "fraction":          kelly_fraction,
        },
    }


if __name__ == "__main__":
    import json
    print("Running backtest (temporal holdout — most recent 30% of data)…")
    result = run_backtest()
    print(json.dumps(result, indent=2))
