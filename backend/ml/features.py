"""
Feature engineering for match outcome, over/under, and BTTS prediction.

Feature vector (34 features):
  Form (6):        home_form5, home_form10, away_form5, away_form10, form_diff, form_momentum
  Goals (8):       home_gf, home_ga, away_gf, away_ga, home_gf_home, home_ga_home,
                   away_gf_away, away_ga_away
  H2H (3):         h2h_home_rate, h2h_draw_rate, h2h_away_rate
  Standing (5):    home_pos, away_pos, home_ppg, away_ppg, pos_diff
  Derived (4):     ppg_diff, total_goals_avg, goal_diff_avg, clean_sheet_rate
  Rest (2):        home_days_rest, away_days_rest
  xG (2):          home_xg_for, away_xg_for   (0.0 if unavailable)
  ELO (1):         elo_diff (home minus away, clamped ±600)
  Attack (3):      home_scoring_std, away_scoring_std, fixture_congestion

NOTE: Bookmaker odds are intentionally excluded from features. Including them
causes the model to replicate market consensus instead of finding independent
edge — which is counterproductive for value betting.

form5/form10 are exponentially weighted (recent matches count more).
"""
import math
import numpy as np
from datetime import date
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
    """Exponentially weighted form — recent matches count more (decay=0.85)."""
    recent = [m for m in matches if m.get("status") == "FINISHED"][-n:]
    if not recent:
        return 0.0
    decay = 0.85
    total_weight = 0.0
    weighted_pts = 0.0
    for i, m in enumerate(recent):
        # earlier matches have lower weight
        w = decay ** (len(recent) - 1 - i)
        r = _result_for_team(m, team_id)
        pts = 3 if r == "W" else (1 if r == "D" else 0)
        weighted_pts += w * pts
        total_weight += w
    return weighted_pts / total_weight if total_weight else 0.0


def _form_momentum(matches: list[dict], team_id: int) -> float:
    """Difference in weighted form: last 3 minus previous 3."""
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


def _days_since_last_match(matches: list[dict], ref_date: Optional[str]) -> float:
    """Days between ref_date and the team's most recent finished match. Capped at 21."""
    if not ref_date:
        return 7.0
    finished = [m for m in matches if m.get("status") == "FINISHED" and m.get("utcDate")]
    if not finished:
        return 7.0
    try:
        last_date_str = max(m["utcDate"][:10] for m in finished)
        d1 = date.fromisoformat(ref_date[:10])
        d2 = date.fromisoformat(last_date_str)
        days = (d1 - d2).days
        return float(min(max(days, 1), 21))
    except Exception:
        return 7.0


def _scoring_std(matches: list[dict], team_id: int, n: int = 10) -> float:
    """Standard deviation of goals scored in last N matches. High = inconsistent scorer."""
    recent = [m for m in matches if m.get("status") == "FINISHED"][-n:]
    goals = []
    for m in recent:
        home_id = m.get("homeTeam", {}).get("id")
        hg = m.get("score", {}).get("fullTime", {}).get("home")
        ag = m.get("score", {}).get("fullTime", {}).get("away")
        if hg is None or ag is None:
            continue
        goals.append(hg if team_id == home_id else ag)
    if len(goals) < 2:
        return 1.0  # neutral default
    mean = sum(goals) / len(goals)
    variance = sum((g - mean) ** 2 for g in goals) / len(goals)
    return float(math.sqrt(variance))


def _fixture_congestion(home_matches: list[dict], away_matches: list[dict],
                        ref_date: Optional[str], window_days: int = 14) -> float:
    """Combined count of matches both teams played in last `window_days` days.
    Higher = more fatigue. Useful signal for mid-week / cup fixture squeezes."""
    if not ref_date:
        return 4.0  # typical: 2 matches each
    try:
        ref = date.fromisoformat(ref_date[:10])
    except Exception:
        return 4.0

    def _count(matches):
        c = 0
        for m in matches:
            if m.get("status") != "FINISHED":
                continue
            try:
                md = date.fromisoformat(m["utcDate"][:10])
                if 0 < (ref - md).days <= window_days:
                    c += 1
            except Exception:
                pass
        return c

    return float(_count(home_matches) + _count(away_matches))


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
    match_date: Optional[str] = None,
    # ELO signal (home_elo - away_elo, clamped to ±600). 0.0 when unavailable.
    elo_diff: float = 0.0,
    # NOTE: bookmaker odds deliberately excluded — see module docstring.
) -> np.ndarray:
    """
    Returns a 1-D float32 numpy feature vector (34 features).
    Feature order MUST stay consistent with training — do not reorder.
    """
    # --- Form (exponentially weighted) ---
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
    pos_diff      = home_pos - away_pos
    ppg_diff      = home_ppg - away_ppg
    total_goals   = home_gf + away_ga
    goal_diff_avg = home_gf - home_ga
    cs_rate       = _clean_sheet_rate(home_matches, home_id)

    # --- Rest / fatigue ---
    home_days_rest = _days_since_last_match(home_matches, match_date)
    away_days_rest = _days_since_last_match(away_matches, match_date)

    # --- Attack consistency & fixture load ---
    home_scoring_std  = _scoring_std(home_matches, home_id)
    away_scoring_std  = _scoring_std(away_matches, away_id)
    fixture_cong      = _fixture_congestion(home_matches, away_matches, match_date)

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
        # Rest (2)
        home_days_rest, away_days_rest,
        # xG (2)
        home_xg, away_xg,
        # ELO (1) — home ELO minus away ELO, clamped ±600
        elo_diff,
        # Attack consistency + fixture load (3)
        home_scoring_std, away_scoring_std, fixture_cong,
    ], dtype=np.float32)


FEATURE_NAMES = [
    # Form (6)
    "home_form5", "home_form10", "away_form5", "away_form10",
    "form_diff", "home_momentum",
    # Goals (8)
    "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
    "home_gf_home", "home_ga_home", "away_gf_away", "away_ga_away",
    # H2H (3)
    "h2h_home_rate", "h2h_draw_rate", "h2h_away_rate",
    # Standing (5)
    "home_position", "away_position", "home_ppg", "away_ppg", "pos_diff",
    # Derived (4)
    "ppg_diff", "total_goals_avg", "goal_diff_avg", "clean_sheet_rate",
    # Rest (2)
    "home_days_rest", "away_days_rest",
    # xG (2)
    "home_xg_for", "away_xg_for",
    # ELO (1)
    "elo_diff",
    # Attack consistency + fixture load (3)
    "home_scoring_std", "away_scoring_std", "fixture_congestion",
]

N_FEATURES = len(FEATURE_NAMES)  # 34
