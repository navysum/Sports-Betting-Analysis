"""
Feature engineering for match outcome, over/under, and BTTS prediction.

Feature vector (28 features):
  Form (6):        home_form5, home_form10, away_form5, away_form10, form_diff, form_momentum
  Goals (8):       home_gf, home_ga, away_gf, away_ga, home_gf_home, home_ga_home,
                   away_gf_away, away_ga_away
  H2H (3):         h2h_home_rate, h2h_draw_rate, h2h_away_rate
  Standing (5):    home_pos, away_pos, home_ppg, away_ppg, pos_diff
  Derived (4):     ppg_diff, total_goals_avg, goal_diff_avg, clean_sheet_rate
  xG (2):          home_xg_for, away_xg_for   (0.0 if unavailable — model handles gracefully)
"""
import numpy as np
from typing import Optional


def _result_for_team(match: dict, team_id: int) -> Optional[str]:
    home_id = match.get("homeTeam", {}).get("id")
    away_id = match.get("awayTeam", {}).get("id")
    hg = match.get("score", {}).get("fullTime", {}).get("home")
    ag = match.get("score", {}).get("fullTime", {}).get("away")
    if hg is None or ag is None:
        return None
    if hg == ag:
        return "D"
    if team_id == home_id:
        return "W" if hg > ag else "L"
    if team_id == away_id:
        return "W" if ag > hg else "L"
    return None


def _goals_for_against(matches: list[dict], team_id: int) -> tuple[float, float]:
    gf, ga, count = 0, 0, 0
    for m in matches:
        home_id = m.get("homeTeam", {}).get("id")
        hg = m.get("score", {}).get("fullTime", {}).get("home")
        ag = m.get("score", {}).get("fullTime", {}).get("away")
        if hg is None or ag is None:
            continue
        if team_id == home_id:
            gf += hg; ga += ag
        else:
            gf += ag; ga += hg
        count += 1
    return (gf / count, ga / count) if count else (0.0, 0.0)


def _form_points(matches: list[dict], team_id: int, n: int = 5) -> float:
    recent = [m for m in matches if m.get("status") == "FINISHED"][-n:]
    pts = sum(3 if _result_for_team(m, team_id) == "W" else (1 if _result_for_team(m, team_id) == "D" else 0) for m in recent)
    return pts / max(len(recent), 1)


def _form_momentum(matches: list[dict], team_id: int) -> float:
    """Difference in form points: last 3 minus previous 3."""
    finished = [m for m in matches if m.get("status") == "FINISHED"]
    last3 = _form_points(finished[-3:], team_id, 3) if len(finished) >= 3 else 0.0
    prev3 = _form_points(finished[-6:-3], team_id, 3) if len(finished) >= 6 else 0.0
    return last3 - prev3


def _clean_sheet_rate(matches: list[dict], team_id: int, n: int = 10) -> float:
    recent = [m for m in matches if m.get("status") == "FINISHED"][-n:]
    cs = 0
    for m in recent:
        home_id = m.get("homeTeam", {}).get("id")
        hg = m.get("score", {}).get("fullTime", {}).get("home")
        ag = m.get("score", {}).get("fullTime", {}).get("away")
        if hg is None or ag is None:
            continue
        goals_conceded = ag if team_id == home_id else hg
        if goals_conceded == 0:
            cs += 1
    return cs / max(len(recent), 1)


def _h2h_stats(h2h_matches: list[dict], home_id: int, away_id: int) -> tuple[float, float, float]:
    results = []
    for m in h2h_matches[-10:]:
        h = m.get("homeTeam", {}).get("id")
        hg = m.get("score", {}).get("fullTime", {}).get("home")
        ag = m.get("score", {}).get("fullTime", {}).get("away")
        if hg is None or ag is None:
            continue
        if h == home_id:
            results.append("H" if hg > ag else ("D" if hg == ag else "A"))
        elif h == away_id:
            results.append("A" if hg > ag else ("D" if hg == ag else "H"))
    n = max(len(results), 1)
    return results.count("H") / n, results.count("D") / n, results.count("A") / n


def build_feature_vector(
    home_id: int,
    away_id: int,
    home_matches: list[dict],
    away_matches: list[dict],
    h2h_matches: list[dict],
    home_standing: Optional[dict] = None,
    away_standing: Optional[dict] = None,
    home_xg: float = 0.0,
    away_xg: float = 0.0,
) -> np.ndarray:
    """
    Returns a 1-D float32 numpy feature vector.
    Feature order MUST stay consistent with training — do not reorder.
    """
    # --- Form ---
    hf5  = _form_points(home_matches, home_id, 5)
    hf10 = _form_points(home_matches, home_id, 10)
    af5  = _form_points(away_matches, away_id, 5)
    af10 = _form_points(away_matches, away_id, 10)
    form_diff = hf5 - af5
    home_momentum = _form_momentum(home_matches, home_id)

    # --- Goals (overall) ---
    home_gf, home_ga = _goals_for_against(home_matches, home_id)
    away_gf, away_ga = _goals_for_against(away_matches, away_id)

    # --- Goals (home/away splits) ---
    home_home_matches = [m for m in home_matches if m.get("homeTeam", {}).get("id") == home_id]
    away_away_matches = [m for m in away_matches if m.get("awayTeam", {}).get("id") == away_id]
    home_gf_h, home_ga_h = _goals_for_against(home_home_matches, home_id)
    away_gf_a, away_ga_a = _goals_for_against(away_away_matches, away_id)

    # --- H2H ---
    h2h_h, h2h_d, h2h_a = _h2h_stats(h2h_matches, home_id, away_id)

    # --- Standings ---
    def _pos(s): return s.get("position", 10) if s else 10
    def _ppg(s): return s.get("points", 0) / max(s.get("playedGames", 1), 1) if s else 0.0

    home_pos = _pos(home_standing)
    away_pos = _pos(away_standing)
    home_ppg = _ppg(home_standing)
    away_ppg = _ppg(away_standing)

    # --- Derived ---
    pos_diff       = home_pos - away_pos
    ppg_diff       = home_ppg - away_ppg
    total_goals    = home_gf + away_ga
    goal_diff_avg  = home_gf - home_ga
    cs_rate        = _clean_sheet_rate(home_matches, home_id)

    return np.array([
        # Form (6)
        hf5, hf10, af5, af10, form_diff, home_momentum,
        # Goals (8)
        home_gf, home_ga, away_gf, away_ga,
        home_gf_h, home_ga_h, away_gf_a, away_ga_a,
        # H2H (3)
        h2h_h, h2h_d, h2h_a,
        # Standing (5)
        home_pos, away_pos, home_ppg, away_ppg, pos_diff,
        # Derived (4)
        ppg_diff, total_goals, goal_diff_avg, cs_rate,
        # xG (2)
        home_xg, away_xg,
    ], dtype=np.float32)


FEATURE_NAMES = [
    # Form
    "home_form5", "home_form10", "away_form5", "away_form10",
    "form_diff", "home_momentum",
    # Goals
    "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
    "home_gf_home", "home_ga_home", "away_gf_away", "away_ga_away",
    # H2H
    "h2h_home_rate", "h2h_draw_rate", "h2h_away_rate",
    # Standing
    "home_position", "away_position", "home_ppg", "away_ppg", "pos_diff",
    # Derived
    "ppg_diff", "total_goals_avg", "goal_diff_avg", "clean_sheet_rate",
    # xG
    "home_xg_for", "away_xg_for",
]

N_FEATURES = len(FEATURE_NAMES)  # 28
