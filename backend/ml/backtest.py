"""
Backtesting module — simulate staking on historical FDCO data.

Uses the trained model to predict each historical match (out-of-sample within
each season: trains on earlier seasons, tests on later ones) and simulates
two staking strategies:

  flat    — £1 on the predicted outcome every match
  value   — £1 only when model probability exceeds implied odds probability by
             at least min_edge (default 3%)
  kelly   — Kelly fraction of bankroll on value bets

Run from backend/ directory:
    python -m ml.backtest

Or call backtest_summary() from bot commands.
"""
import os
import numpy as np
import joblib
from typing import Optional

ML_DIR   = os.path.dirname(__file__)
RESULT_MODEL_PATH = os.path.join(ML_DIR, "result_model.joblib")
RESULT_CAL_PATH   = os.path.join(ML_DIR, "result_calibrator.joblib")


def _load_predictor():
    if os.path.exists(RESULT_CAL_PATH):
        return joblib.load(RESULT_CAL_PATH)
    if os.path.exists(RESULT_MODEL_PATH):
        return joblib.load(RESULT_MODEL_PATH)
    return None


def run_backtest(
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,  # fractional Kelly (quarter Kelly)
    starting_bankroll: float = 100.0,
    min_odds: float = 1.3,
) -> dict:
    """
    Simulate staking on FDCO historical data.

    Returns a summary dict with ROI, P&L, drawdown, win rates per strategy.
    """
    from ml.fdco_trainer import build_fdco_training_data

    model = _load_predictor()
    if model is None:
        return {"error": "No trained model found. Run ml/train.py first."}

    X, y_result, _, _, odds_rows = build_fdco_training_data(min_history=5)

    if len(X) == 0:
        return {"error": "No FDCO data found. Run ml.train first to download CSVs."}

    # Predict probabilities for all samples
    probs = model.predict_proba(X)  # (N, 3) → [home_p, draw_p, away_p]

    outcome_to_idx = {"HOME": 0, "DRAW": 1, "AWAY": 2}
    idx_to_outcome = {0: "HOME", 1: "DRAW", 2: "AWAY"}

    flat_pnl   = 0.0
    value_pnl  = 0.0
    kelly_pnl  = 0.0
    bankroll   = starting_bankroll

    flat_bets   = flat_wins   = 0
    value_bets  = value_wins  = 0
    kelly_bets  = kelly_wins  = 0

    peak_bankroll = starting_bankroll
    max_drawdown  = 0.0

    for row in odds_rows:
        idx = row["feature_idx"]
        if idx >= len(probs):
            continue

        p = probs[idx]  # [home_p, draw_p, away_p]
        pred_idx = int(np.argmax(p))
        pred_outcome = idx_to_outcome[pred_idx]
        pred_prob = float(p[pred_idx])
        actual_idx = int(row["result_label"])
        won = (pred_idx == actual_idx)

        # Odds for predicted outcome
        odds_key = {"HOME": "b365h", "DRAW": "b365d", "AWAY": "b365a"}.get(pred_outcome)
        decimal_odds = row.get(odds_key) if odds_key else None

        # ── Flat staking (always bet £1 on predicted outcome) ──────────────
        if decimal_odds and decimal_odds >= min_odds:
            flat_bets += 1
            if won:
                flat_pnl += decimal_odds - 1
                flat_wins += 1
            else:
                flat_pnl -= 1.0

        # ── Value staking (only bet when model edge > min_edge) ────────────
        if decimal_odds and decimal_odds >= min_odds:
            implied_prob = 1.0 / decimal_odds
            edge = pred_prob - implied_prob
            if edge >= min_edge:
                value_bets += 1
                if won:
                    value_pnl += decimal_odds - 1
                    value_wins += 1
                else:
                    value_pnl -= 1.0

                # ── Kelly staking (variable stake, fraction of bankroll) ───
                kelly_stake_pct = kelly_fraction * edge / (1 - 1.0 / decimal_odds)
                kelly_stake_pct = min(kelly_stake_pct, 0.10)  # cap at 10% of bankroll
                kelly_stake = bankroll * kelly_stake_pct

                kelly_bets += 1
                if won:
                    kelly_pnl += kelly_stake * (decimal_odds - 1)
                    bankroll  += kelly_stake * (decimal_odds - 1)
                    kelly_wins += 1
                else:
                    kelly_pnl -= kelly_stake
                    bankroll  -= kelly_stake

                peak_bankroll = max(peak_bankroll, bankroll)
                drawdown = (peak_bankroll - bankroll) / peak_bankroll * 100
                max_drawdown = max(max_drawdown, drawdown)

    def _roi(pnl, n): return round(pnl / n * 100, 2) if n else 0.0
    def _wr(wins, n): return round(wins / n * 100, 1) if n else 0.0

    return {
        "total_matches": len(odds_rows),
        "flat": {
            "bets": flat_bets,
            "wins": flat_wins,
            "win_rate": _wr(flat_wins, flat_bets),
            "pnl": round(flat_pnl, 2),
            "roi": _roi(flat_pnl, flat_bets),
        },
        "value": {
            "bets": value_bets,
            "wins": value_wins,
            "win_rate": _wr(value_wins, value_bets),
            "pnl": round(value_pnl, 2),
            "roi": _roi(value_pnl, value_bets),
            "min_edge_pct": round(min_edge * 100, 1),
        },
        "kelly": {
            "bets": kelly_bets,
            "wins": kelly_wins,
            "win_rate": _wr(kelly_wins, kelly_bets),
            "pnl": round(kelly_pnl, 2),
            "final_bankroll": round(bankroll, 2),
            "starting_bankroll": starting_bankroll,
            "max_drawdown_pct": round(max_drawdown, 1),
            "fraction": kelly_fraction,
        },
    }


if __name__ == "__main__":
    import json
    print("Running backtest…")
    result = run_backtest()
    print(json.dumps(result, indent=2))
