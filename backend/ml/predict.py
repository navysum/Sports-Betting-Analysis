"""
Multi-market prediction service.

Loads XGBoost models + isotonic calibrators at startup. Calibrators produce
better-calibrated probabilities than raw XGBoost outputs — especially important
for value bet edge calculations.

Also blends Dixon-Coles Poisson model (when available and team names are known)
to improve draw pricing and produce correct-score probabilities.

Computes:
  - Star confidence rating (1–5)
  - Value bet flags (where model prob > implied bookmaker prob)
  - Dixon-Coles expected goals and correct-score distribution
"""
import os
import numpy as np
import joblib
from typing import Optional

ML_DIR = os.path.dirname(__file__)

RESULT_MODEL_PATH = os.path.join(ML_DIR, "result_model.joblib")
GOALS_MODEL_PATH  = os.path.join(ML_DIR, "goals_model.joblib")
BTTS_MODEL_PATH   = os.path.join(ML_DIR, "btts_model.joblib")

RESULT_CAL_PATH   = os.path.join(ML_DIR, "result_calibrator.joblib")
GOALS_CAL_PATH    = os.path.join(ML_DIR, "goals_calibrator.joblib")
BTTS_CAL_PATH     = os.path.join(ML_DIR, "btts_calibrator.joblib")

# Legacy path for backward compatibility
LEGACY_MODEL_PATH = os.path.join(ML_DIR, "model.joblib")

_result_model = None
_goals_model  = None
_btts_model   = None
_result_cal   = None
_goals_cal    = None
_btts_cal     = None
_dc_model     = None   # Dixon-Coles model (optional)


def load_model():
    """Load all models and calibrators. Falls back gracefully if absent."""
    global _result_model, _goals_model, _btts_model
    global _result_cal, _goals_cal, _btts_cal, _dc_model

    if os.path.exists(RESULT_MODEL_PATH):
        _result_model = joblib.load(RESULT_MODEL_PATH)
        print("Result model loaded.")
    elif os.path.exists(LEGACY_MODEL_PATH):
        _result_model = joblib.load(LEGACY_MODEL_PATH)
        print("Legacy result model loaded (run ml/train.py to upgrade).")
    else:
        print("WARNING: No result model found. Run ml/train.py first.")

    if os.path.exists(GOALS_MODEL_PATH):
        _goals_model = joblib.load(GOALS_MODEL_PATH)
        print("Goals model loaded.")
    else:
        print("WARNING: No goals model found.")

    if os.path.exists(BTTS_MODEL_PATH):
        _btts_model = joblib.load(BTTS_MODEL_PATH)
        print("BTTS model loaded.")
    else:
        print("WARNING: No BTTS model found.")

    for path, name, var_name in [
        (RESULT_CAL_PATH, "result", "_result_cal"),
        (GOALS_CAL_PATH,  "goals",  "_goals_cal"),
        (BTTS_CAL_PATH,   "btts",   "_btts_cal"),
    ]:
        if os.path.exists(path):
            globals()[var_name] = joblib.load(path)
            print(f"{name.capitalize()} calibrator loaded.")

    # Dixon-Coles model (optional — present after first retrain post-upgrade)
    try:
        from ml.dixon_coles import load_dc_model
        _dc_model = load_dc_model()
        if _dc_model:
            print(f"Dixon-Coles model loaded ({len(_dc_model.attack)} teams).")
        else:
            print("Dixon-Coles model not found (will fit on next retrain).")
    except Exception as e:
        print(f"Dixon-Coles load failed: {e}")


def _proba(model, calibrator, vec: np.ndarray) -> Optional[np.ndarray]:
    """Return calibrated probabilities if calibrator is available, else raw."""
    if calibrator is not None:
        return calibrator.predict_proba(vec)[0]
    if model is not None:
        return model.predict_proba(vec)[0]
    return None


def _star_rating(confidence: float) -> int:
    if confidence >= 0.70: return 5
    if confidence >= 0.60: return 4
    if confidence >= 0.50: return 3
    if confidence >= 0.40: return 2
    return 1


def _is_value(model_prob: float, bookmaker_odds: Optional[float], min_edge: float = 0.03) -> bool:
    if bookmaker_odds is None or bookmaker_odds <= 1.0:
        return False
    implied_prob = 1 / bookmaker_odds
    return model_prob > (implied_prob + min_edge)


def predict(
    feature_vector: np.ndarray,
    bookmaker_odds: Optional[dict] = None,
    home_team: str = "",
    away_team: str = "",
) -> dict:
    """
    Run all available models and return a comprehensive prediction dict.

    bookmaker_odds (optional): {"home": 2.10, "draw": 3.40, "away": 3.20,
                                "over25": 1.85, "btts": 1.75}
    home_team / away_team: used for Dixon-Coles lookup (FDCO team names).

    Returns:
        home_win_prob, draw_prob, away_win_prob, predicted_outcome, confidence,
        stars, over25_prob, btts_prob, over25_predicted, btts_predicted,
        value_bets, calibrated, dc_available, correct_scores, xg_home, xg_away
    """
    vec  = feature_vector.reshape(1, -1)
    odds = bookmaker_odds or {}

    # ── XGBoost result model ──────────────────────────────────────────────────
    result_probs = _proba(_result_model, _result_cal, vec)
    if result_probs is not None:
        home_p = float(result_probs[0])
        draw_p = float(result_probs[1])
        away_p = float(result_probs[2])
    else:
        home_p = draw_p = away_p = 1 / 3

    # ── XGBoost goals + BTTS models ───────────────────────────────────────────
    goals_probs = _proba(_goals_model, _goals_cal, vec)
    over25_prob = float(goals_probs[1]) if goals_probs is not None else 0.5

    btts_probs = _proba(_btts_model, _btts_cal, vec)
    btts_prob  = float(btts_probs[1]) if btts_probs is not None else 0.5

    # ── Dixon-Coles blend ─────────────────────────────────────────────────────
    dc_info = None
    correct_scores: list = []
    xg_home = xg_away = None

    if _dc_model and home_team and away_team:
        from ml.dixon_coles import DC_BLEND_WEIGHT
        dc_info = _dc_model.match_probs(home_team, away_team)
        if dc_info:
            w_dc  = DC_BLEND_WEIGHT
            w_xgb = 1.0 - DC_BLEND_WEIGHT

            home_p    = w_xgb * home_p    + w_dc * dc_info["home"]
            draw_p    = w_xgb * draw_p    + w_dc * dc_info["draw"]
            away_p    = w_xgb * away_p    + w_dc * dc_info["away"]
            over25_prob = w_xgb * over25_prob + w_dc * dc_info["over25"]
            btts_prob   = w_xgb * btts_prob   + w_dc * dc_info["btts"]

            # Renormalise result probs to sum to 1
            total = home_p + draw_p + away_p
            if total > 0:
                home_p /= total; draw_p /= total; away_p /= total

            correct_scores = dc_info.get("correct_scores", [])
            xg_home = dc_info.get("xg_home")
            xg_away = dc_info.get("xg_away")

    # ── Final outcome ─────────────────────────────────────────────────────────
    pred_idx = int(np.argmax([home_p, draw_p, away_p]))
    outcome_map = {0: "HOME", 1: "DRAW", 2: "AWAY"}
    predicted_outcome = outcome_map[pred_idx]
    confidence = round(max(home_p, draw_p, away_p), 4)
    stars = _star_rating(confidence)

    # ── Value bets ────────────────────────────────────────────────────────────
    value_bets = []
    if _is_value(home_p, odds.get("home")):
        value_bets.append(f"Home Win (model {home_p:.0%} vs implied {1/odds['home']:.0%})")
    if _is_value(draw_p, odds.get("draw")):
        value_bets.append(f"Draw (model {draw_p:.0%} vs implied {1/odds['draw']:.0%})")
    if _is_value(away_p, odds.get("away")):
        value_bets.append(f"Away Win (model {away_p:.0%} vs implied {1/odds['away']:.0%})")
    if _is_value(over25_prob, odds.get("over25")):
        value_bets.append(f"Over 2.5 (model {over25_prob:.0%} vs implied {1/odds['over25']:.0%})")
    if _is_value(btts_prob, odds.get("btts")):
        value_bets.append(f"BTTS Yes (model {btts_prob:.0%} vs implied {1/odds['btts']:.0%})")

    return {
        "home_win_prob":     round(home_p, 4),
        "draw_prob":         round(draw_p, 4),
        "away_win_prob":     round(away_p, 4),
        "predicted_outcome": predicted_outcome,
        "confidence":        confidence,
        "stars":             stars,
        "over25_prob":       round(over25_prob, 4),
        "btts_prob":         round(btts_prob, 4),
        "over25_predicted":  over25_prob >= 0.5,
        "btts_predicted":    btts_prob >= 0.5,
        "value_bets":        value_bets,
        "calibrated":        _result_cal is not None,
        "dc_available":      dc_info is not None,
        "correct_scores":    correct_scores,
        "xg_home":           xg_home,
        "xg_away":           xg_away,
    }
