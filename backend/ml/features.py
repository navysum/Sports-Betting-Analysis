"""
Feature engineering for match outcome, over/under, and BTTS prediction.

Feature vector (46 features):
  Form (7):        home_form5, home_form10, away_form5, away_form10, form_diff,
                   home_momentum, away_momentum
                   form_diff = home_form5_home − away_form5_away (venue-specific)
  Goals (8):       home_gf, home_ga, away_gf, away_ga, home_gf_home, home_ga_home,
                   away_gf_away, away_ga_away
  H2H (3):         h2h_home_rate, h2h_draw_rate, h2h_away_rate
  Standing (5):    home_pos_norm, away_pos_norm, home_ppg, away_ppg, pos_diff_norm
  Derived (5):     ppg_diff, total_goals_avg, goal_diff_avg, away_goal_diff_avg,
                   clean_sheet_rate
  Rest (2):        home_days_rest, away_days_rest
  xG (4):          home_xg_for, away_xg_for, home_xg_against, away_xg_against
                   (training: SOT × 0.27 proxy; inference: Understat actual xG)
  ELO (1):         elo_diff (home minus away, clamped ±600)
  Attack (3):      home_scoring_std, away_scoring_std, fixture_congestion
  Venue form (4):  home_form5_home, away_form5_away,
                   home_goal_diff_home, away_goal_diff_away
                   (venue-specific form — home team at home, away team on road)
  Context (2):     season_stage, away_clean_sheet_rate
  Draw (2):        home_draw_rate, away_draw_rate

NOTE: N_FEATURES changed from 44 → 46 when draw-propensity features were
  activated. Shots features (always 0.0) replaced by venue form in v2.
  The training_cache.npz is automatically discarded on feature version mismatch
  (see _load_cache()). Always trigger a retrain after changing features.

NOTE: Bookmaker odds are intentionally excluded from features. Including them
causes the model to replicate market consensus instead of finding independent
edge — which is counterproductive for value betting.

NOTE: xG features are populated with SOT×0.27 proxy in training (FDCO CSVs
have shots data), and real Understat xG at inference when available.

form5/form10 are exponentially weighted (recent matches count more).
Positions are normalised to [0,1] by total teams in the league (default 20).
"""
import math
import numpy as np
from datetime import date
from typing import Optional

# Increment when feature semantics change so cached training data is discarded.
FEATURE_VERSION = 2


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
    """Exponentially weighted form — recent matches count more.

    Decay = 0.80 (previously 0.85): weights the 5th-most-recent match at
    ~41% of the most recent, vs 52% at 0.85. More sensitive to recent form
    change; reduces season-crossover contamination.
    """
    recent = [m for m in matches if m.get("status") == "FINISHED"][-n:]
    if not recent:
        return 0.0
    decay = 0.80
    total_weight = 0.0
    weighted_pts = 0.0
    for i, m in enumerate(recent):
        w = decay ** (len(recent) - 1 - i)
        r = _result_for_team(m, team_id)
        pts = 3 if r == "W" else (1 if r == "D" else 0)
        weighted_pts += w * pts
        total_weight += w
    return weighted_pts / total_weight if total_weight else 0.0


def _form_momentum(matches: list[dict], team_id: int) -> float:
    """Difference in weighted form: last 3 minus previous 3.

    FIX #20: filter finished matches once here; _form_points internally does
    the same filter but on these already-filtered slices — harmless but
    previously redundant. Pre-filtering here removes the wasted pass.
    """
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
    """Return H2H win rates for (home, draw, away).

    Returns a neutral prior of (1/3, 1/3, 1/3) when no H2H data is available.
    Previously returned (0, 0, 0) which was indistinguishable from a perfectly
    balanced history and caused silent bias on first-time matchups or newly
    promoted clubs.
    """
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
    if not results:
        # Neutral prior — no H2H data available (new fixture / promoted club)
        return 1 / 3, 1 / 3, 1 / 3
    n = len(results)
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


def _draw_rate(matches: list[dict], team_id: int, n: int = 10) -> float:
    """
    Fraction of last n finished matches that ended in a draw.

    Draw propensity is a team-level signal: some clubs (typically mid-table,
    defensively organised, with limited attacking quality) draw far more often
    than their form points or goal ratios suggest. This feature gives the model
    a direct handle on that tendency, improving calibration of draw probabilities.

    Returns 0.0 if no finished matches are available (safe default — neutral).
    """
    recent = [m for m in matches if m.get("status") == "FINISHED"][-n:]
    if not recent:
        return 0.0
    draws = 0
    for m in recent:
        hg = m.get("score", {}).get("fullTime", {}).get("home")
        ag = m.get("score", {}).get("fullTime", {}).get("away")
        if hg is not None and ag is not None and hg == ag:
            draws += 1
    return draws / len(recent)


def _shots_sot_avg(matches: list[dict], team_id: int, n: int = 10) -> tuple[float, float]:
    """Rolling average shots and shots-on-target over last n finished matches."""
    recent = [m for m in matches if m.get("status") == "FINISHED"][-n:]
    shots_list, sot_list = [], []
    for m in recent:
        home_id = m.get("homeTeam", {}).get("id")
        shots = m.get("shots")
        if not shots:
            continue
        if team_id == home_id:
            shots_list.append(shots.get("home") or 0)
            sot_list.append(shots.get("homeSot") or 0)
        else:
            shots_list.append(shots.get("away") or 0)
            sot_list.append(shots.get("awaySot") or 0)
    shots_avg = sum(shots_list) / len(shots_list) if shots_list else 0.0
    sot_avg   = sum(sot_list)   / len(sot_list)   if sot_list   else 0.0
    return shots_avg, sot_avg


def _season_stage(match_date: Optional[str]) -> float:
    """
    Season progress normalised to [0, 1].
    European football season: Aug (0.0) through May (1.0).
    Returns 0.5 if date is unavailable or month is off-season.
    """
    if not match_date:
        return 0.5
    try:
        month = int(match_date[5:7])
        season_months = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
        if month in season_months:
            return season_months.index(month) / (len(season_months) - 1)
        return 0.5
    except Exception:
        return 0.5


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
        return 1.0
    mean = sum(goals) / len(goals)
    # FIX #19: sample variance (N-1 denominator) — population variance (N) understated
    # variance by up to 20% on 5–10 game windows.
    variance = sum((g - mean) ** 2 for g in goals) / (len(goals) - 1)
    return float(math.sqrt(variance))


def _fixture_congestion(home_matches: list[dict], away_matches: list[dict],
                        ref_date: Optional[str], window_days: int = 14) -> float:
    """Combined count of matches both teams played in last `window_days` days."""
    if not ref_date:
        return 4.0
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
    home_xg_against: float = 0.0,
    away_xg_against: float = 0.0,
    match_date: Optional[str] = None,
    elo_diff: float = 0.0,
    # Total teams in the league — used to normalise standing position to [0,1].
    # Typical values: PL/PD/SA/FL1=20, BL1/DED=18, ELC=24. Default 20.
    total_teams: int = 20,
) -> np.ndarray:
    """
    Returns a 1-D float32 numpy feature vector (46 features).
    Feature order MUST stay consistent with training — do not reorder.
    """
    # --- Venue-filtered match lists (shared by form, goals, and venue features) ---
    home_home_matches = [m for m in home_matches if m.get("homeTeam", {}).get("id") == home_id]
    away_away_matches = [m for m in away_matches if m.get("awayTeam", {}).get("id") == away_id]

    # --- Venue-specific form (computed early — used in form_diff below) ---
    home_form5_home = _form_points(home_home_matches, home_id, 5)
    away_form5_away = _form_points(away_away_matches, away_id, 5)

    # --- Form (exponentially weighted) ---
    hf5  = _form_points(home_matches, home_id, 5)
    hf10 = _form_points(home_matches, home_id, 10)
    af5  = _form_points(away_matches, away_id, 5)
    af10 = _form_points(away_matches, away_id, 10)
    # Venue-specific form diff: home team at home vs away team on the road.
    # More predictive than overall form_diff because venue context dominates.
    form_diff      = home_form5_home - away_form5_away
    home_momentum  = _form_momentum(home_matches, home_id)
    away_momentum  = _form_momentum(away_matches, away_id)

    # --- Goals (overall) ---
    home_gf, home_ga = _goals_for_against(home_matches, home_id)
    away_gf, away_ga = _goals_for_against(away_matches, away_id)

    # --- Goals (home/away splits) ---
    home_gf_h, home_ga_h = _goals_for_against(home_home_matches, home_id)
    away_gf_a, away_ga_a = _goals_for_against(away_away_matches, away_id)
    away_cs_rate = _clean_sheet_rate(away_away_matches, away_id)

    # --- Venue-specific goal diff (GF − GA per game at this venue) ---
    home_goal_diff_home = home_gf_h - home_ga_h
    away_goal_diff_away = away_gf_a - away_ga_a

    # --- H2H ---
    h2h_h, h2h_d, h2h_a = _h2h_stats(h2h_matches, home_id, away_id)

    # --- Standings (normalised position: 1st place → 1/total, last → 1.0) ---
    def _pos(s): return s.get("position", 10) if s else 10
    def _ppg(s): return s.get("points", 0) / max(s.get("playedGames", 1), 1) if s else 0.0

    denom = max(total_teams, 1)
    home_pos_norm = _pos(home_standing) / denom
    away_pos_norm = _pos(away_standing) / denom
    home_ppg      = _ppg(home_standing)
    away_ppg      = _ppg(away_standing)

    # --- Derived ---
    pos_diff_norm  = home_pos_norm - away_pos_norm
    ppg_diff       = home_ppg - away_ppg
    # FIX #4: correct expected-goals-total formula.
    # Old: home_gf + away_ga  → double-counted one side (inflated for attack vs defensive mismatches).
    # New: average of both teams' offensive + defensive rates gives a symmetric estimate of
    # expected total goals: E[total] ≈ (home_gf + home_ga + away_gf + away_ga) / 2.
    total_goals    = (home_gf + home_ga + away_gf + away_ga) / 2.0
    goal_diff_avg  = home_gf - home_ga        # home team's GD per game
    away_goal_diff_avg = away_gf - away_ga    # away team's GD per game
    cs_rate        = _clean_sheet_rate(home_matches, home_id)

    # --- Rest / fatigue ---
    home_days_rest = _days_since_last_match(home_matches, match_date)
    away_days_rest = _days_since_last_match(away_matches, match_date)

    # --- Attack consistency & fixture load ---
    home_scoring_std = _scoring_std(home_matches, home_id)
    away_scoring_std = _scoring_std(away_matches, away_id)
    fixture_cong     = _fixture_congestion(home_matches, away_matches, match_date)

    # --- Season stage & context ---
    s_stage = _season_stage(match_date)

    # --- Draw propensity (Audit Fix #11) ---
    home_draw_rate = _draw_rate(home_matches, home_id)
    away_draw_rate = _draw_rate(away_matches, away_id)

    return np.array([
        # Form (7)
        hf5, hf10, af5, af10, form_diff, home_momentum, away_momentum,
        # Goals (8)
        home_gf, home_ga, away_gf, away_ga,
        home_gf_h, home_ga_h, away_gf_a, away_ga_a,
        # H2H (3)
        h2h_h, h2h_d, h2h_a,
        # Standing (5) — positions normalised by league size
        home_pos_norm, away_pos_norm, home_ppg, away_ppg, pos_diff_norm,
        # Derived (5)
        ppg_diff, total_goals, goal_diff_avg, away_goal_diff_avg, cs_rate,
        # Rest (2)
        home_days_rest, away_days_rest,
        # xG (4) — SOT×0.27 proxy in training, Understat values at inference
        home_xg, away_xg, home_xg_against, away_xg_against,
        # ELO (1)
        elo_diff,
        # Attack consistency + fixture load (3)
        home_scoring_std, away_scoring_std, fixture_cong,
        # Venue-specific form (4) — home team at home, away team on the road
        home_form5_home, away_form5_away, home_goal_diff_home, away_goal_diff_away,
        # Context (2)
        s_stage, away_cs_rate,
        # Draw propensity (2) — Audit Fix #11
        home_draw_rate, away_draw_rate,
    ], dtype=np.float32)


FEATURE_NAMES = [
    # Form (7)
    "home_form5", "home_form10", "away_form5", "away_form10",
    "form_diff", "home_momentum", "away_momentum",
    # Goals (8)
    "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
    "home_gf_home", "home_ga_home", "away_gf_away", "away_ga_away",
    # H2H (3)
    "h2h_home_rate", "h2h_draw_rate", "h2h_away_rate",
    # Standing (5)
    "home_pos_norm", "away_pos_norm", "home_ppg", "away_ppg", "pos_diff_norm",
    # Derived (5)
    "ppg_diff", "total_goals_avg", "goal_diff_avg", "away_goal_diff_avg", "clean_sheet_rate",
    # Rest (2)
    "home_days_rest", "away_days_rest",
    # xG (4)
    "home_xg_for", "away_xg_for", "home_xg_against", "away_xg_against",
    # ELO (1)
    "elo_diff",
    # Attack consistency + fixture load (3)
    "home_scoring_std", "away_scoring_std", "fixture_congestion",
    # Venue-specific form (4)
    "home_form5_home", "away_form5_away", "home_goal_diff_home", "away_goal_diff_away",
    # Context (2)
    "season_stage", "away_clean_sheet_rate",
    # Draw propensity (2) — Audit Fix #11
    "home_draw_rate", "away_draw_rate",
]

N_FEATURES = len(FEATURE_NAMES)  # 46
