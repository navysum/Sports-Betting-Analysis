"""
Blend weight optimiser for Dixon-Coles / XGBoost ensemble.

Searches for the per-market DC blend weight that minimises Brier score on a
holdout set of FDCO historical matches. Saves the result to data/blend_weights.json.

Usage:
    cd backend
    python -m ml.optimize_blend

Output:
    Prints best weights per market and saves to data/blend_weights.json.

How it works:
  For each market (result / over25 / btts / over35) we try DC blend weights
  in 0.05 steps from 0.0 to 1.0 and pick the weight that minimises Brier
  score on the last 20% of FDCO data (chronological split — no leakage).
"""
import json
import os
import sys

# Allow running from backend/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BLEND_PATH = os.path.join(DATA_DIR, "blend_weights.json")


def _brier_binary(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def _brier_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score for a 3-class problem (result market)."""
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def optimise():
    from ml.fdco_trainer import build_fdco_training_data
    from ml.dixon_coles import load_dc_model

    print("Loading FDCO training data…")
    X, y_result, y_goals, y_btts, y_over35, odds_rows, _ = build_fdco_training_data()
    if len(X) == 0:
        print("No training data found. Run scraper first.")
        return

    dc_model = load_dc_model()
    if dc_model is None:
        print("Dixon-Coles model not found. Run a full retrain first.")
        return

    import joblib
    ml_dir = os.path.dirname(__file__)

    def _load(name):
        path = os.path.join(ml_dir, name)
        return joblib.load(path) if os.path.exists(path) else None

    result_cal  = _load("result_calibrator.joblib")  or _load("result_model.joblib")
    goals_cal   = _load("goals_calibrator.joblib")   or _load("goals_model.joblib")
    btts_cal    = _load("btts_calibrator.joblib")     or _load("btts_model.joblib")
    over35_cal  = _load("over35_calibrator.joblib")   or _load("over35_model.joblib")

    if any(m is None for m in [result_cal, goals_cal, btts_cal, over35_cal]):
        print("One or more models missing. Run a full retrain first.")
        return

    # Chronological holdout: last 20% of samples
    n = len(X)
    split = int(n * 0.80)
    X_hold      = X[split:]
    y_res_hold  = y_result[split:]
    y_ou_hold   = y_goals[split:]
    y_bt_hold   = y_btts[split:]
    y_o35_hold  = y_over35[split:]
    rows_hold   = odds_rows[split:]

    print(f"Holdout size: {len(X_hold)} samples ({len(X_hold)/n:.0%} of total)")

    # XGBoost probabilities on holdout
    xgb_result  = result_cal.predict_proba(X_hold)          # (N, 3)
    xgb_over25  = goals_cal.predict_proba(X_hold)[:, 1]     # (N,)
    xgb_btts    = btts_cal.predict_proba(X_hold)[:, 1]
    xgb_over35  = over35_cal.predict_proba(X_hold)[:, 1]

    # DC probabilities on holdout
    dc_home, dc_draw, dc_away = [], [], []
    dc_over25, dc_btts, dc_over35 = [], [], []

    for row in rows_hold:
        # FIX #8: pass league so per-league rho is used during blend evaluation
        info = dc_model.match_probs(row["home"], row["away"], league=row.get("league", ""))
        if info:
            dc_home.append(info["home"])
            dc_draw.append(info["draw"])
            dc_away.append(info["away"])
            dc_over25.append(info.get("over25", 0.5))
            dc_btts.append(info.get("btts", 0.5))
            dc_over35.append(info.get("over35", 0.35))
        else:
            dc_home.append(1/3); dc_draw.append(1/3); dc_away.append(1/3)
            dc_over25.append(0.5); dc_btts.append(0.5); dc_over35.append(0.35)

    dc_result = np.column_stack([dc_home, dc_draw, dc_away])
    dc_over25 = np.array(dc_over25)
    dc_btts   = np.array(dc_btts)
    dc_over35 = np.array(dc_over35)

    best = {}
    # FIX #24: 0.01 step grid (was 0.05) — optimal weight could be 0.27, which
    # a 0.05 grid rounded to 0.25 or 0.30, costing measurable Brier score.
    weights = np.arange(0.0, 1.01, 0.01)

    markets = [
        ("result", xgb_result, dc_result, y_res_hold, "multiclass"),
        ("over25", xgb_over25, dc_over25, y_ou_hold,  "binary"),
        ("btts",   xgb_btts,   dc_btts,   y_bt_hold,  "binary"),
        ("over35", xgb_over35, dc_over35, y_o35_hold, "binary"),
    ]

    for name, xgb_p, dc_p, labels, kind in markets:
        best_w, best_score = 0.5, float("inf")
        for w in weights:
            blended = (1 - w) * xgb_p + w * dc_p
            if kind == "binary":
                score = _brier_binary(blended, labels)
            else:
                # Renormalise result probs
                totals = blended.sum(axis=1, keepdims=True)
                blended = blended / np.where(totals > 0, totals, 1)
                score = _brier_multiclass(blended, labels)
            if score < best_score:
                best_score = score
                best_w = float(w)
        best[name] = round(best_w, 2)
        print(f"  {name:6s}  best DC weight = {best_w:.2f}  (Brier = {best_score:.4f})")

    print(f"\nOptimal blend weights: {best}")
    with open(BLEND_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Saved to {BLEND_PATH}")


if __name__ == "__main__":
    optimise()
