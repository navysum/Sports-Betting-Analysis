"""
Multi-market prediction service.

Loads XGBoost models + isotonic calibrators at startup. Calibrators produce
better-calibrated probabilities than raw XGBoost outputs — especially important
for value bet edge calculations.

Also blends Dixon-Coles Poisson/NB model (when available and team names are
known) to improve draw pricing and produce correct-score probabilities.

Fixes applied:
  FIX #6  — Per-league calibrators are now cached in a module-level dict after
             the first load. Previously they were deserialized from disk on every
             single predict() call — 4 × joblib.load() per prediction.
  FIX #14 — Value detection now uses devigged (margin-free) implied probabilities
             for 1X2 markets. Raw bookmaker odds include a 4–8% overround, so
             comparing model prob vs raw implied understated every edge figure.
             For binary markets (over25, btts, over35) we use the Pinnacle
             no-vig convention: if both sides are present we devig exactly;
             otherwise we assume a 5% binary margin.
  FIX #8  — DC match_probs() call now passes league_code so the per-league rho
             τ-correction is used instead of the global average rho.
"""
import json
import os
import numpy as np
import joblib
from typing import Optional

ML_DIR = os.path.dirname(__file__)
_BLEND_WEIGHTS_PATH = os.path.join(ML_DIR, "..", "data", "blend_weights.json")
_DEFAULT_BLEND = {"result": 0.50, "over25": 0.50, "btts": 0.50, "over35": 0.50}


def _load_blend_weights() -> dict:
    try:
        with open(_BLEND_WEIGHTS_PATH, encoding="utf-8") as f:
            w = json.load(f)
        return {k: float(w.get(k, _DEFAULT_BLEND[k])) for k in _DEFAULT_BLEND}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULT_BLEND)


RESULT_MODEL_PATH  = os.path.join(ML_DIR, "result_model.joblib")
GOALS_MODEL_PATH   = os.path.join(ML_DIR, "goals_model.joblib")
BTTS_MODEL_PATH    = os.path.join(ML_DIR, "btts_model.joblib")
OVER35_MODEL_PATH  = os.path.join(ML_DIR, "over35_model.joblib")

RESULT_CAL_PATH    = os.path.join(ML_DIR, "result_calibrator.joblib")
GOALS_CAL_PATH     = os.path.join(ML_DIR, "goals_calibrator.joblib")
BTTS_CAL_PATH      = os.path.join(ML_DIR, "btts_calibrator.joblib")
OVER35_CAL_PATH    = os.path.join(ML_DIR, "over35_calibrator.joblib")

LEGACY_MODEL_PATH  = os.path.join(ML_DIR, "model.joblib")

_result_model  = None
_goals_model   = None
_btts_model    = None
_over35_model  = None
_result_cal    = None
_goals_cal     = None
_btts_cal      = None
_over35_cal    = None
_dc_model      = None

# FIX #6: module-level caches so per-league models/calibrators are loaded from
# disk at most once per server lifetime instead of on every predict() call.
_league_model_cache: dict[str, Optional[object]] = {}
_league_cal_cache:   dict[str, Optional[object]] = {}


def load_model():
    """Load all models and calibrators. Falls back gracefully if absent."""
    global _result_model, _goals_model, _btts_model, _over35_model
    global _result_cal, _goals_cal, _btts_cal, _over35_cal, _dc_model

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

    if os.path.exists(OVER35_MODEL_PATH):
        _over35_model = joblib.load(OVER35_MODEL_PATH)
        print("Over35 model loaded.")
    else:
        print("WARNING: No over35 model found (retrain to generate).")

    for path, name, var_name in [
        (RESULT_CAL_PATH, "result",  "_result_cal"),
        (GOALS_CAL_PATH,  "goals",   "_goals_cal"),
        (BTTS_CAL_PATH,   "btts",    "_btts_cal"),
        (OVER35_CAL_PATH, "over35",  "_over35_cal"),
    ]:
        if os.path.exists(path):
            globals()[var_name] = joblib.load(path)
            print(f"{name.capitalize()} calibrator loaded.")

    try:
        from ml.dixon_coles import load_dc_model
        _dc_model = load_dc_model()
        if _dc_model:
            print(f"Dixon-Coles model loaded ({len(_dc_model.attack)} teams, "
                  f"r_nb={_dc_model.r_nb:.1f}, "
                  f"{len(_dc_model.rho_by_league)} league rhos).")
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


def _load_league_model(base_path: str, league_code: str) -> Optional[object]:
    """
    FIX #6: cached load of a league-specific model.
    Returns None if absent — caller falls back to the combined model.
    """
    if not league_code:
        return None
    cache_key = f"{base_path}|{league_code}"
    if cache_key in _league_model_cache:
        return _league_model_cache[cache_key]

    league_path = base_path.replace(".joblib", f"_{league_code}.joblib")
    result = None
    if os.path.exists(league_path):
        try:
            result = joblib.load(league_path)
        except Exception:
            pass
    _league_model_cache[cache_key] = result
    return result


def _load_league_calibrator(base_cal_path: str, league_code: str) -> Optional[object]:
    """
    FIX #6: cached load of a league-specific calibrator.
    Previously called joblib.load() on every single predict() invocation;
    now loaded at most once per league per server lifetime.
    """
    if not league_code:
        return None
    cache_key = f"{base_cal_path}|{league_code}"
    if cache_key in _league_cal_cache:
        return _league_cal_cache[cache_key]

    league_cal_path = base_cal_path.replace(".joblib", f"_{league_code}.joblib")
    result = None
    if os.path.exists(league_cal_path):
        try:
            result = joblib.load(league_cal_path)
        except Exception:
            pass
    _league_cal_cache[cache_key] = result
    return result


# ─── Devigging helpers (FIX #14) ─────────────────────────────────────────────

def _devig_1x2(
    home_odds: Optional[float],
    draw_odds: Optional[float],
    away_odds: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Remove bookmaker margin from 1X2 decimal odds.

    Returns fair implied probabilities (sum to 1.0) or (None, None, None) if
    any price is missing or invalid.

    Without devigging a 4–8% overround causes every edge figure to be
    understated — a model at 52% vs fair 50% shows as no value against
    vig-inflated 54.1% implied.
    """
    if not all(o and o > 1.0 for o in [home_odds, draw_odds, away_odds]):
        return None, None, None
    total = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)
    if total <= 0:
        return None, None, None
    return (1 / home_odds) / total, (1 / draw_odds) / total, (1 / away_odds) / total


def _devig_binary(
    odds_a: Optional[float],
    odds_b: Optional[float] = None,
) -> Optional[float]:
    """
    Remove overround from one side of a binary market.

    If both sides are provided (e.g. over25 + under25) devigging is exact.
    If only one side is available we assume a 5% binary margin:
      fair_implied ≈ raw_implied / 1.025
    """
    if not odds_a or odds_a <= 1.0:
        return None
    raw = 1.0 / odds_a
    if odds_b and odds_b > 1.0:
        total = raw + (1.0 / odds_b)
        return raw / total
    # single-sided approximation (5% binary margin)
    return min(max(raw / 1.025, 0.01), 0.99)


def _star_rating(value_bets: list, best_edge: float, confidence: float) -> int:
    """
    Stars reflect betting quality (edge over market), not raw win probability.
      5★ = large edge (≥15%) or very high confidence (≥72%)
      4★ = solid edge (≥10%) or high confidence (≥62%)
      3★ = moderate edge (≥7%) or decent confidence (≥52%)
      2★ = value detected (≥5% edge) or marginal confidence (≥42%)
      1★ = no value detected (below 5% min_edge threshold)

    Bottom threshold raised from 3% to 5% to match the min_edge filter change.
    """
    if value_bets:
        if best_edge >= 0.15: return 5
        if best_edge >= 0.10: return 4
        if best_edge >= 0.07: return 3
        return 2  # ≥5% (already passing the value filter)
    if confidence >= 0.72: return 5
    if confidence >= 0.62: return 4
    if confidence >= 0.52: return 3
    if confidence >= 0.42: return 2
    return 1


def _is_value(model_prob: float, fair_implied: Optional[float], min_edge: float = 0.05) -> bool:
    """Compare model probability vs fair (devigged) implied probability.

    Threshold raised 3% → 5%: tighter filter reduces false positives and better
    separates real edge from noise. Aligned with backtest.py's min_edge default.
    """
    if fair_implied is None:
        return False
    return model_prob > (fair_implied + min_edge)


def _kelly_fraction(model_prob: float, bookmaker_odds: float, fraction: float = 0.25) -> float:
    """Quarter-Kelly stake as a fraction of bankroll. Returns 0.0 if no edge."""
    if not bookmaker_odds or bookmaker_odds <= 1.0:
        return 0.0
    b = bookmaker_odds - 1.0
    q = 1.0 - model_prob
    kelly = (b * model_prob - q) / b
    if kelly <= 0:
        return 0.0
    return round(min(kelly * fraction, 0.05), 4)


def predict(
    feature_vector: np.ndarray,
    bookmaker_odds: Optional[dict] = None,
    home_team: str = "",
    away_team: str = "",
    league_code: str = "",
) -> dict:
    """
    Run all available models and return a comprehensive prediction dict.

    bookmaker_odds (optional): {"home": 2.10, "draw": 3.40, "away": 3.20,
                                "over25": 1.85, "under25": 1.95,
                                "btts": 1.75, "over35": 2.80}
    home_team / away_team: FDCO team names for Dixon-Coles lookup.
    league_code: competition code (e.g. "PL") for per-league rho (FIX #8).
    """
    vec  = feature_vector.reshape(1, -1)
    odds = bookmaker_odds or {}

    # ── Per-league model + calibrator (FIX #6: cached loads) ─────────────────
    active_result_model = _load_league_model(RESULT_MODEL_PATH, league_code) or _result_model
    active_goals_model  = _load_league_model(GOALS_MODEL_PATH,  league_code) or _goals_model
    active_btts_model   = _load_league_model(BTTS_MODEL_PATH,   league_code) or _btts_model
    active_over35_model = _load_league_model(OVER35_MODEL_PATH, league_code) or _over35_model

    active_result_cal  = _load_league_calibrator(RESULT_CAL_PATH,  league_code) or _result_cal
    active_goals_cal   = _load_league_calibrator(GOALS_CAL_PATH,   league_code) or _goals_cal
    active_btts_cal    = _load_league_calibrator(BTTS_CAL_PATH,    league_code) or _btts_cal
    active_over35_cal  = _load_league_calibrator(OVER35_CAL_PATH,  league_code) or _over35_cal

    # ── XGBoost result model ──────────────────────────────────────────────────
    result_probs = _proba(active_result_model, active_result_cal, vec)
    if result_probs is not None:
        home_p = float(result_probs[0])
        draw_p = float(result_probs[1])
        away_p = float(result_probs[2])
    else:
        home_p = draw_p = away_p = 1 / 3

    goals_probs  = _proba(active_goals_model,  active_goals_cal,  vec)
    over25_prob  = float(goals_probs[1])  if goals_probs  is not None else 0.5
    btts_probs   = _proba(active_btts_model,   active_btts_cal,   vec)
    btts_prob    = float(btts_probs[1])   if btts_probs   is not None else 0.5
    over35_probs = _proba(active_over35_model, active_over35_cal, vec)
    over35_prob  = float(over35_probs[1]) if over35_probs is not None else 0.35

    # ── Dixon-Coles blend (FIX #8: pass league for per-league rho) ───────────
    dc_info = None
    correct_scores: list = []
    xg_home = xg_away = None
    score_grid = score_grid_size = dc_rho = dc_r_nb = None

    if _dc_model and home_team and away_team:
        dc_info = _dc_model.match_probs(home_team, away_team, league=league_code)
        if dc_info:
            bw = _load_blend_weights()

            w_res = bw["result"];  w_xgb_res = 1.0 - w_res
            w_ou  = bw["over25"]; w_xgb_ou  = 1.0 - w_ou
            w_bt  = bw["btts"];   w_xgb_bt  = 1.0 - w_bt
            w_o35 = bw["over35"]; w_xgb_o35 = 1.0 - w_o35

            home_p    = w_xgb_res * home_p    + w_res * dc_info["home"]
            draw_p    = w_xgb_res * draw_p    + w_res * dc_info["draw"]
            away_p    = w_xgb_res * away_p    + w_res * dc_info["away"]
            over25_prob = w_xgb_ou * over25_prob + w_ou * dc_info["over25"]
            btts_prob   = w_xgb_bt * btts_prob   + w_bt * dc_info["btts"]
            if "over35" in dc_info:
                over35_prob = w_xgb_o35 * over35_prob + w_o35 * dc_info["over35"]

            total = home_p + draw_p + away_p
            if total > 0:
                home_p /= total; draw_p /= total; away_p /= total

            correct_scores  = dc_info.get("correct_scores", [])
            xg_home         = dc_info.get("xg_home")
            xg_away         = dc_info.get("xg_away")
            score_grid      = dc_info.get("score_grid")
            score_grid_size = dc_info.get("score_grid_size", 9)
            dc_rho          = dc_info.get("rho")
            dc_r_nb         = dc_info.get("r_nb")

    # Fallback xG: when DC can't look up a team (name mismatch between the API
    # and FDCO CSV training data), xg_home would be None and the Monte Carlo page
    # would filter out every match showing "No matches with xG data available".
    # Derive lambda from the blended over25 probability instead:
    #   lambda_total ≈ -1.5 × ln(1 − P(over25))   [good for P ∈ 0.4–0.85]
    # Split 58.5 / 41.5 home/away to reflect typical home-advantage goal share.
    used_xg_fallback = xg_home is None
    if xg_home is None:
        import math as _math
        _lam = max(0.5, -1.5 * _math.log(max(1.0 - over25_prob, 0.01)))
        xg_home = round(_lam * 0.585, 2)
        xg_away = round(_lam * 0.415, 2)

    # ── Devigged fair implied probabilities (FIX #14) ─────────────────────────
    home_fair, draw_fair, away_fair = _devig_1x2(
        odds.get("home"), odds.get("draw"), odds.get("away")
    )
    over25_fair = _devig_binary(odds.get("over25"), odds.get("under25"))
    over35_fair = _devig_binary(odds.get("over35"), odds.get("under35"))
    btts_fair   = _devig_binary(odds.get("btts"),   odds.get("btts_no"))

    # Exact devig available when all three 1X2 prices are present
    used_approx_devig = not all(o and o > 1.0 for o in [
        odds.get("home"), odds.get("draw"), odds.get("away")
    ])
    # League model used if a league-specific file exists on disk
    league_model_active = bool(league_code and os.path.exists(
        RESULT_MODEL_PATH.replace(".joblib", f"_{league_code}.joblib")
    ))
    used_global_model = not league_model_active
    used_dc_fallback  = dc_info is None

    # ── Value bets (FIX #14: compare vs fair implied, not raw implied) ────────
    value_bets   = []
    kelly_stakes = {}
    edges        = {}

    checks = [
        ("home",   home_p,      home_fair,   odds.get("home"),   "Home Win"),
        ("draw",   draw_p,      draw_fair,   odds.get("draw"),   "Draw"),
        ("away",   away_p,      away_fair,   odds.get("away"),   "Away Win"),
        ("over25", over25_prob, over25_fair, odds.get("over25"), "Over 2.5"),
        ("over35", over35_prob, over35_fair, odds.get("over35"), "Over 3.5"),
        ("btts",   btts_prob,   btts_fair,   odds.get("btts"),   "BTTS Yes"),
    ]

    for key, prob, fair_implied, book_odds, label in checks:
        if _is_value(prob, fair_implied):
            edge = prob - fair_implied
            edges[key] = edge
            k = _kelly_fraction(prob, book_odds)
            kelly_stakes[key] = k
            raw_implied = (1 / book_odds) if book_odds and book_odds > 1 else None
            value_bets.append(
                f"{label} (model {prob:.0%} vs fair {fair_implied:.0%}"
                + (f" / vig {raw_implied:.0%}" if raw_implied else "")
                + f", edge +{edge:.0%}, Kelly {k:.1%})"
            )

    best_edge = max(edges.values()) if edges else 0.0

    # ── Final outcome ─────────────────────────────────────────────────────────
    pred_idx = int(np.argmax([home_p, draw_p, away_p]))
    outcome_map = {0: "HOME", 1: "DRAW", 2: "AWAY"}
    predicted_outcome = outcome_map[pred_idx]
    confidence = round(max(home_p, draw_p, away_p), 4)
    stars = _star_rating(value_bets, best_edge, confidence)

    # ── No-bet detector — explicit AVOID signals ──────────────────────────────
    # Distinct from the eligibility gate: AVOID means the model has a specific
    # negative signal (edge is wrong direction, too many quality failures, or
    # confidence is near-uniform). PASS just means "not good enough to bet".
    avoid_reasons: list[str] = []
    if best_edge < 0:
        avoid_reasons.append(
            f"Model probability below fair odds (edge {best_edge * 100:.1f}% — market has the edge)"
        )
    if sum(bool(v) for v in [used_xg_fallback, used_dc_fallback, used_global_model, used_approx_devig]) >= 3:
        avoid_reasons.append("Three or more data quality flags active — prediction reliability low")
    if confidence < 0.52:
        avoid_reasons.append("Probabilities near-uniform (confidence < 52%) — insufficient signal")

    # ── Bet eligibility gate ───────────────────────────────────────────────────
    # Five conditions must ALL be true before a bet surfaces as a recommendation.
    # The data-quality trio (devig, xG, DC) filters low-information predictions;
    # edge and confidence together filter low-value ones. Changing either
    # threshold here affects what surfaces on the Best Bets page.
    bet_eligible = all([
        not used_approx_devig,      # exact two-sided devig required
        not used_xg_fallback,       # real xG source (not shots-on-target proxy)
        not used_dc_fallback,       # Dixon-Coles converged for this fixture
        not used_global_model,      # league-specific model (not global fallback)
        best_edge >= 0.05,          # ≥ 5% edge over the fair implied probability
        confidence >= 0.55,         # ≥ 55% model confidence on the predicted outcome
    ])

    return {
        "home_win_prob":     round(home_p, 4),
        "draw_prob":         round(draw_p, 4),
        "away_win_prob":     round(away_p, 4),
        "predicted_outcome": predicted_outcome,
        "confidence":        confidence,
        "stars":             stars,
        "over25_prob":       round(over25_prob, 4),
        "over35_prob":       round(over35_prob, 4),
        "btts_prob":         round(btts_prob, 4),
        "over25_predicted":  over25_prob >= 0.5,
        "over35_predicted":  over35_prob >= 0.5,
        "btts_predicted":    btts_prob >= 0.5,
        "value_bets":        value_bets,
        "kelly_stakes":      kelly_stakes,
        "best_edge":         round(best_edge, 4),
        "edges":             {k: round(v, 4) for k, v in edges.items()},
        "calibrated":        _result_cal is not None,
        "dc_available":      dc_info is not None,
        "correct_scores":    correct_scores,
        "score_grid":        score_grid,
        "score_grid_size":   score_grid_size,
        "dc_rho":            dc_rho,
        "dc_r_nb":           dc_r_nb,
        "xg_home":           xg_home,
        "xg_away":           xg_away,
        "bookmaker_odds":    dict(odds) if odds else None,
        "league_model_used": league_model_active,
        "league_cal_used": bool(league_code and os.path.exists(
            RESULT_CAL_PATH.replace(".joblib", f"_{league_code}.joblib")
        )),
        # Fallback flags — logged for quality monitoring
        "fallback_flags": {
            "used_xg_fallback":   used_xg_fallback,
            "used_dc_fallback":   used_dc_fallback,
            "used_global_model":  used_global_model,
            "used_approx_devig":  used_approx_devig,
        },
        "bet_eligible": bet_eligible,
        "avoid_signal": {
            "should_avoid": len(avoid_reasons) > 0,
            "reasons":      avoid_reasons,
        },
    }
