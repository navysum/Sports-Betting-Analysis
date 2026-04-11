"""
Feature engineering for match outcome, over/under, and BTTS prediction.

Feature vector (41 features):
  Form (7):        home_form5, home_form10, away_form5, away_form10, form_diff,
                   home_momentum, away_momentum
  Goals (8):       home_gf, home_ga, away_gf, away_ga, home_gf_home, home_ga_home,
                   away_gf_away, away_ga_away
  H2H (3):         h2h_home_rate, h2h_draw_rate, h2h_away_rate
  Standing (5):    home_pos, away_pos, home_ppg, away_ppg, pos_diff
  Derived (5):     ppg_diff, home_cs_rate, away_cs_rate,
                   home_cs_rate_home, away_cs_rate_away
  Shots (4):       home_sot_avg, away_sot_avg, home_shot_accuracy, away_shot_accuracy
                   (0.0 when shot data unavailable — API inference, older seasons)
  Rest (2):        home_days_rest, away_days_rest
  xG (2):          home_xg_for, away_xg_for   (0.0 if unavailable)
  ELO (1):         elo_diff   (home_elo - away_elo, clamped ±600)
  Market (3):      fair_home_prob, fair_draw_prob, fair_away_prob
                   (overround-removed implied probs; 0.0 when odds unavailable)
  League (1):      league_id  (0-7 per league, 8 for unknown/European)

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


def _clean_sheet_rate(
    matches: list[dict], team_id: int, n: int = 10, venue: str = "all"
) -> float:
    """
    Fraction of recent matches in which the team kept a clean sheet.

    venue: "all" (default) | "home" (only home games) | "away" (only away games)
    """
    finished = [m for m in matches if m.get("status") == "FINISHED"]
    if venue == "home":
        finished = [m for m in finished if m.get("homeTeam", {}).get("id") == team_id]
    elif venue == "away":
        finished = [m for m in finished if m.get("awayTeam", {}).get("id") == team_id]
    recent = finished[-n:]
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


def _shots_on_target_avg(matches: list[dict], team_id: int, n: int = 5) -> float:
    """
    Rolling average shots on target per game over the last N finished matches.
    Uses _hst / _ast fields injected by fdco_trainer. Returns 0.0 when absent
    (API-based matches don't carry shot data).
    """
    finished = [m for m in matches if m.get("status") == "FINISHED"][-n:]
    total, count = 0.0, 0
    for m in finished:
        home_id = m.get("homeTeam", {}).get("id")
        sot = m.get("_hst") if team_id == home_id else m.get("_ast")
        if sot is not None:
            total += float(sot)
            count += 1
    return total / count if count else 0.0


def _shot_accuracy(matches: list[dict], team_id: int, n: int = 5) -> float:
    """
    Rolling shots-on-target / total shots ratio over the last N matches.
    Returns 0.0 when shot data is absent.
    """
    finished = [m for m in matches if m.get("status") == "FINISHED"][-n:]
    total_shots, total_sot = 0.0, 0.0
    for m in finished:
        home_id = m.get("homeTeam", {}).get("id")
        if team_id == home_id:
            s, sot = m.get("_hs"), m.get("_hst")
        else:
            s, sot = m.get("_as"), m.get("_ast")
        if s is not None and sot is not None and float(s) > 0:
            total_shots += float(s)
            total_sot   += float(sot)
    return total_sot / total_shots if total_shots > 0 else 0.0


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


def _fair_implied_probs(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """
    Convert decimal bookmaker odds to fair implied probabilities by removing
    the overround (bookmaker margin).

    E.g. 2.10 / 3.40 / 3.20  →  raw implied 0.476 / 0.294 / 0.313 (sums to 1.083)
                                  fair probs  0.440 / 0.272 / 0.289 (sums to 1.000)

    Returns (0.0, 0.0, 0.0) when any odds are missing or implausible.
    """
    if not home_odds or not draw_odds or not away_odds:
        return 0.0, 0.0, 0.0
    if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
        return 0.0, 0.0, 0.0
    rh = 1.0 / home_odds
    rd = 1.0 / draw_odds
    ra = 1.0 / away_odds
    total = rh + rd + ra
    if total <= 0:
        return 0.0, 0.0, 0.0
    return rh / total, rd / total, ra / total


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
    # Bookmaker decimal odds. 0.0 when unavailable. Converted to fair implied probs.
    home_odds: float = 0.0,
    draw_odds: float = 0.0,
    away_odds: float = 0.0,
    # League identifier. 0-7 for known leagues, -1/8 for unknown/European.
    league_id: int = -1,
) -> np.ndarray:
    """
    Returns a 1-D float32 numpy feature vector (35 features).
    Feature order MUST stay consistent with training — do not reorder.
    """
    # --- Form (exponentially weighted) ---
    hf5  = _form_points(home_matches, home_id, 5)
    hf10 = _form_points(home_matches, home_id, 10)
    af5  = _form_points(away_matches, away_id, 5)
    af10 = _form_points(away_matches, away_id, 10)
    form_diff     = hf5 - af5
    home_momentum = _form_momentum(home_matches, home_id)
    away_momentum = _form_momentum(away_matches, away_id)

    # --- Goals (overall) ---
    home_gf, home_ga = _goals_for_against(home_matches, home_id)
    away_gf, away_ga = _goals_for_against(away_matches, away_id)

    # --- Goals (home/away venue splits) ---
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
    pos_diff           = home_pos - away_pos
    ppg_diff           = home_ppg - away_ppg
    home_cs_rate       = _clean_sheet_rate(home_matches, home_id)
    away_cs_rate       = _clean_sheet_rate(away_matches, away_id)
    # Venue-split: home team's defensive record at home, away team's record away
    home_cs_rate_home  = _clean_sheet_rate(home_matches, home_id, venue="home")
    away_cs_rate_away  = _clean_sheet_rate(away_matches, away_id, venue="away")

    # --- Shots (0.0 when shot data absent — API inference, older FDCO seasons) ---
    home_sot     = _shots_on_target_avg(home_matches, home_id)
    away_sot     = _shots_on_target_avg(away_matches, away_id)
    home_shot_acc = _shot_accuracy(home_matches, home_id)
    away_shot_acc = _shot_accuracy(away_matches, away_id)

    # --- Rest / fatigue ---
    home_days_rest = _days_since_last_match(home_matches, match_date)
    away_days_rest = _days_since_last_match(away_matches, match_date)

    # --- Market: convert decimal odds → fair implied probabilities ---
    fair_home, fair_draw, fair_away = _fair_implied_probs(home_odds, draw_odds, away_odds)

    # --- League ID (8 = unknown/European) ---
    league_feat = float(league_id) if league_id >= 0 else 8.0

    return np.array([
        # Form (7)
        hf5, hf10, af5, af10, form_diff, home_momentum, away_momentum,
        # Goals (8)
        home_gf, home_ga, away_gf, away_ga,
        home_gf_h, home_ga_h, away_gf_a, away_ga_a,
        # H2H (3)
        h2h_h, h2h_d, h2h_a,
        # Standing (5)
        home_pos, away_pos, home_ppg, away_ppg, pos_diff,
        # Derived (5) — ppg_diff + overall + venue-split clean sheet rates
        ppg_diff, home_cs_rate, away_cs_rate, home_cs_rate_home, away_cs_rate_away,
        # Shots (4) — 0.0 when unavailable
        home_sot, away_sot, home_shot_acc, away_shot_acc,
        # Rest (2)
        home_days_rest, away_days_rest,
        # xG (2)
        home_xg, away_xg,
        # ELO (1)
        elo_diff,
        # Market — fair implied probs, 0.0 when unavailable (3)
        fair_home, fair_draw, fair_away,
        # League (1)
        league_feat,
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
    "home_position", "away_position", "home_ppg", "away_ppg", "pos_diff",
    # Derived (5) — overall + venue-split clean sheet rates
    "ppg_diff", "home_cs_rate", "away_cs_rate",
    "home_cs_rate_home", "away_cs_rate_away",
    # Shots (4) — 0.0 when unavailable at inference time
    "home_sot_avg", "away_sot_avg", "home_shot_accuracy", "away_shot_accuracy",
    # Rest (2)
    "home_days_rest", "away_days_rest",
    # xG (2)
    "home_xg_for", "away_xg_for",
    # ELO (1)
    "elo_diff",
    # Market — fair implied probabilities (3)
    "fair_home_prob", "fair_draw_prob", "fair_away_prob",
    # League (1)
    "league_id",
]

N_FEATURES = len(FEATURE_NAMES)  # 41
