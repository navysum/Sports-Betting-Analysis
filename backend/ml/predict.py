"""
Multi-market prediction service.

Loads result, goals, and BTTS models once at startup and exposes predict().
Also computes:
  - Star confidence rating (1–5)
  - Value bet flags (where model prob > implied bookmaker prob)
"""
import os
import numpy as np
import joblib
from typing import Optional

ML_DIR = os.path.dirname(__file__)
RESULT_MODEL_PATH = os.path.join(ML_DIR, "result_model.joblib")
GOALS_MODEL_PATH  = os.path.join(ML_DIR, "goals_model.joblib")
BTTS_MODEL_PATH   = os.path.join(ML_DIR, "btts_model.joblib")

# Legacy path for backward compatibility
LEGACY_MODEL_PATH = os.path.join(ML_DIR, "model.joblib")

_result_model = None
_goals_model  = None
_btts_model   = None


def load_model():
    """Load all three models. Falls back to legacy single model if available."""
    global _result_model, _goals_model, _btts_model

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
        print("WARNING: No goals model found. Run ml/train.py first.")

    if os.path.exists(BTTS_MODEL_PATH):
        _btts_model = joblib.load(BTTS_MODEL_PATH)
        print("BTTS model loaded.")
    else:
        print("WARNING: No BTTS model found. Run ml/train.py first.")


def _star_rating(confidence: float) -> int:
    """Convert probability confidence to 1–5 stars."""
    if confidence >= 0.70: return 5
    if confidence >= 0.60: return 4
    if confidence >= 0.50: return 3
    if confidence >= 0.40: return 2
    return 1


def _is_value(model_prob: float, bookmaker_odds: Optional[float], min_edge: float = 0.03) -> bool:
    """True if model probability exceeds implied bookmaker probability by at least min_edge."""
    if bookmaker_odds is None or bookmaker_odds <= 1.0:
        return False
    implied_prob = 1 / bookmaker_odds
    return model_prob > (implied_prob + min_edge)


def predict(
    feature_vector: np.ndarray,
    bookmaker_odds: Optional[dict] = None,
) -> dict:
    """
    Run all available models and return a comprehensive prediction dict.

    bookmaker_odds (optional): {"home": 2.10, "draw": 3.40, "away": 3.20,
                                "over25": 1.85, "btts": 1.75}

    Returns:
        home_win_prob, draw_prob, away_win_prob, predicted_outcome, confidence,
        stars, over25_prob, btts_prob, over25_predicted, btts_predicted,
        value_bets (list of market strings)
    """
    vec = feature_vector.reshape(1, -1)
    odds = bookmaker_odds or {}

    # --- Result model ---
    if _result_model is not None:
        probs = _result_model.predict_proba(vec)[0]
        home_p, draw_p, away_p = float(probs[0]), float(probs[1]), float(probs[2])
    else:
        home_p = draw_p = away_p = 1 / 3

    pred_idx = int(np.argmax([home_p, draw_p, away_p]))
    outcome_map = {0: "HOME", 1: "DRAW", 2: "AWAY"}
    predicted_outcome = outcome_map[pred_idx]
    confidence = round(max(home_p, draw_p, away_p), 4)
    stars = _star_rating(confidence)

    # --- Goals model (Over/Under 2.5) ---
    if _goals_model is not None:
        g_probs = _goals_model.predict_proba(vec)[0]
        over25_prob = round(float(g_probs[1]), 4)   # class 1 = Over
    else:
        over25_prob = 0.5

    # --- BTTS model ---
    if _btts_model is not None:
        b_probs = _btts_model.predict_proba(vec)[0]
        btts_prob = round(float(b_probs[1]), 4)     # class 1 = Yes
    else:
        btts_prob = 0.5

    # --- Value bets ---
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
        "home_win_prob": round(home_p, 4),
        "draw_prob":     round(draw_p, 4),
        "away_win_prob": round(away_p, 4),
        "predicted_outcome": predicted_outcome,
        "confidence":    confidence,
        "stars":         stars,
        "over25_prob":   over25_prob,
        "btts_prob":     btts_prob,
        "over25_predicted": over25_prob >= 0.5,
        "btts_predicted":   btts_prob >= 0.5,
        "value_bets":    value_bets,
    }
