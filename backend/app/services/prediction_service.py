"""
Orchestrates data fetching, feature engineering, prediction, and ledger logging
for a single match or a batch of upcoming matches.

New signals applied as post-processing adjustments (on top of model output):
  - Injury/suspension counts (API-Football) — reduces win prob for affected team
  - xPts vs actual Pts differential (FBref) — regression-to-mean signal
  - xGD from FBref — supplements Understat xG for quality assessment

CLV tracking: every prediction is logged with opening Pinnacle odds so we can
measure whether the model consistently beats the closing line over time.
"""
import asyncio
import time
from datetime import datetime
from typing import Optional

from app.services.football_api import (
    get_team_matches,
    get_standings,
    get_h2h,
    SUPPORTED_COMPETITIONS,
)
from app.services.scraper import fetch_understat_team_xg
from app.services.rapidapi_football import get_sofascore_team_xg
from app.services.evaluator import append_prediction, build_ledger_entry
from app.services.odds_api import find_match_odds, find_pinnacle_odds
from app.services.injury_service import get_team_injuries, injury_adjustment
# FIX #11: FBref scraper removed from the prediction hot-path.
# get_team_xg_stats() scraped FBref on every prediction request — it typically
# took 2–4 s, raised for rate-limit blocks, and the xPts adjustment it powered
# has been disabled (see _xpts_adjustment comment below). The import is kept as
# a commented reference so it can be re-enabled if an independent backtest
# validates the xPts signal in the future.
# from app.services.fbref_scraper import get_team_xg_stats
from app.services.clv_tracker import log_prediction
from app.utils.team_names import resolve as resolve_team
from ml.features import build_feature_vector
from ml.predict import predict
from ml.elo import load_elo_ratings, EloSystem

# FIX #12: TTL-based ELO refresh.
# ELO ratings are trained once and cached on disk. When a retrain runs on the
# server (e.g. nightly cron), the in-memory _elo object is stale until the next
# process restart. We reload from disk every hour so the live system picks up
# updated ratings without requiring a full restart.
_ELO_TTL: float = 3600.0          # seconds between disk reloads

_elo: EloSystem       = load_elo_ratings()
_elo_loaded_at: float = time.monotonic()


def _get_elo() -> EloSystem:
    """Return the module-level ELO system, refreshing from disk if the TTL has expired."""
    global _elo, _elo_loaded_at
    if time.monotonic() - _elo_loaded_at > _ELO_TTL:
        _elo           = load_elo_ratings()
        _elo_loaded_at = time.monotonic()
    return _elo

# FIX #3 — xG scale mismatch between training and inference.
# During training (FDCO CSVs), xG features are computed as: sot_rolling_avg × 0.27
# Typical sot_avg ≈ 4.0 shots/game → training xG ≈ 1.08 per team.
# At inference, Understat real xG is used — typical top-league team ≈ 1.30–1.50/game,
# which is 20–40% higher than the training distribution. Without rescaling, the
# model sees out-of-distribution xG values and mispredicts attacking-team matchups.
#
# Scale factor: (mean training proxy) / (mean real xG) ≈ 1.08 / 1.35 ≈ 0.80
# This maps real Understat xG back into the range the model was trained on.
_XG_INFERENCE_SCALE: float = 0.80


async def _safe(coro, default=None):
    try:
        return await coro
    except Exception:
        return default if default is not None else []


def _competition_name(code: str) -> str:
    return SUPPORTED_COMPETITIONS.get(code, code)


def _make_match_id(league: str, date: str, home: str, away: str) -> str:
    def _abbr(name: str) -> str:
        return "".join(w[:3].upper() for w in name.split()[:2])
    return f"{league}-{date}-{_abbr(home)}-{_abbr(away)}"


def _apply_prob_adjustments(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    home_adj: float,
    away_adj: float,
) -> tuple[float, float, float]:
    """
    Apply additive adjustments to win probabilities and renormalise.

    Positive adj = boost that team's win prob.
    Negative adj = reduce that team's win prob.
    Draw absorbs the remainder to keep the sum at 1.0.
    """
    home_prob = max(0.01, home_prob + home_adj)
    away_prob = max(0.01, away_prob + away_adj)
    draw_prob = max(0.01, 1.0 - home_prob - away_prob)
    total = home_prob + draw_prob + away_prob
    return home_prob / total, draw_prob / total, away_prob / total


def _xpts_adjustment(stats: dict) -> float:
    """
    Convert FBref xPts-vs-actual differential to a probability nudge.

    Teams overperforming their xPts are due for regression; underperformers
    are due to improve. We apply a small correction to avoid recency bias.

    pts_over_xpts > 0: team is lucky → small negative adj
    pts_over_xpts < 0: team is unlucky → small positive adj

    Capped at ±0.04 per team.
    """
    pox = stats.get("pts_over_xpts", 0.0)
    if pox is None:
        return 0.0
    # -0.008 per extra point/game vs expected, capped at ±0.04
    return max(-0.04, min(0.04, -pox * 0.008))


async def predict_match(
    home_team_id: int,
    away_team_id: int,
    competition_code: str = "PL",
    api_match_id: Optional[int] = None,
    home_team_name: str = "",
    away_team_name: str = "",
    match_date: str = "",
    save_to_ledger: bool = False,
    bookmaker_odds: Optional[dict] = None,
) -> dict:
    """
    Full prediction pipeline for one match.

    Fetches all data sources in parallel, runs the XGBoost+DC ensemble,
    applies injury and xPts regression adjustments, then logs CLV context.
    """
    date_str = match_date or datetime.utcnow().strftime("%Y-%m-%d")

    # ── Core data fetch (parallel) ────────────────────────────────────────────
    # FIX #11: FBref calls removed — get_team_xg_stats() was adding 2–4 s of
    # latency on every request and the xPts signal it supplied has been disabled.
    # home_fbref / away_fbref default to empty dicts; the frontend still handles
    # them gracefully (they're optional display fields, not used in probability).
    (
        home_matches,
        away_matches,
        standings_table,
        home_xg_sofascore,
        away_xg_sofascore,
        home_xg_understat,
        away_xg_understat,
        home_injuries,
        away_injuries,
    ) = await asyncio.gather(
        _safe(get_team_matches(home_team_id, limit=25)),
        _safe(get_team_matches(away_team_id, limit=25)),
        _safe(get_standings(competition_code)),
        # Sofascore: primary xG source (real API, no HTML scraping)
        _safe(get_sofascore_team_xg(home_team_name, competition_code), {}),
        _safe(get_sofascore_team_xg(away_team_name, competition_code), {}),
        # Understat: fallback xG source (HTML scraper, slower / less reliable)
        _safe(fetch_understat_team_xg(home_team_name, competition_code), {}),
        _safe(fetch_understat_team_xg(away_team_name, competition_code), {}),
        _safe(get_team_injuries(home_team_name, competition_code, date_str), []),
        _safe(get_team_injuries(away_team_name, competition_code, date_str), []),
    )
    # Prefer Sofascore xG; fall back to Understat if Sofascore returned nothing
    home_xg_data = home_xg_sofascore or home_xg_understat or {}
    away_xg_data = away_xg_sofascore or away_xg_understat or {}

    home_fbref: dict = {}
    away_fbref: dict = {}

    # H2H — only if we have a match ID
    h2h_matches = []
    if api_match_id:
        h2h_matches = await _safe(get_h2h(api_match_id), [])

    # xG enrichment (Understat) — scaled to match training distribution.
    # Raw Understat values are multiplied by _XG_INFERENCE_SCALE (0.80) to bring
    # them into the same range as the SOT×0.27 proxy used during training.
    home_xg         = (home_xg_data or {}).get("last5_xg_for",     0.0) * _XG_INFERENCE_SCALE
    away_xg         = (away_xg_data or {}).get("last5_xg_for",     0.0) * _XG_INFERENCE_SCALE
    home_xg_against = (home_xg_data or {}).get("last5_xg_against", 0.0) * _XG_INFERENCE_SCALE
    away_xg_against = (away_xg_data or {}).get("last5_xg_against", 0.0) * _XG_INFERENCE_SCALE

    standing_map = {row["team"]["id"]: row for row in (standings_table or [])}

    home_fdco = resolve_team(home_team_name) if home_team_name else home_team_name
    away_fdco = resolve_team(away_team_name) if away_team_name else away_team_name

    # FIX #12: use _get_elo() to get TTL-refreshed ELO ratings
    elo_diff = _get_elo().get_diff(home_fdco, away_fdco) if home_fdco else 0.0

    # Fetch best-available odds + Pinnacle odds in parallel
    if bookmaker_odds is None and home_team_name and competition_code:
        bookmaker_odds, pinnacle_odds = await asyncio.gather(
            _safe(find_match_odds(home_team_name, away_team_name, competition_code), None),
            _safe(find_pinnacle_odds(home_team_name, away_team_name, competition_code), None),
        )
    else:
        pinnacle_odds = await _safe(
            find_pinnacle_odds(home_team_name, away_team_name, competition_code), None
        )

    _LEAGUE_SIZES = {
        "PL": 20, "ELC": 24, "PD": 20, "BL1": 18,
        "SA": 20, "FL1": 20, "DED": 18, "PPL": 18,
    }
    total_teams = _LEAGUE_SIZES.get(competition_code, 20)

    vec = build_feature_vector(
        home_id=home_team_id,
        away_id=away_team_id,
        home_matches=home_matches,
        away_matches=away_matches,
        h2h_matches=h2h_matches,
        home_standing=standing_map.get(home_team_id),
        away_standing=standing_map.get(away_team_id),
        home_xg=home_xg,
        away_xg=away_xg,
        home_xg_against=home_xg_against,
        away_xg_against=away_xg_against,
        match_date=date_str or None,
        elo_diff=elo_diff,
        total_teams=total_teams,
    )

    result = predict(
        vec,
        bookmaker_odds=bookmaker_odds,
        home_team=home_fdco,
        away_team=away_fdco,
        league_code=competition_code,
    )

    # ── Post-processing adjustments ───────────────────────────────────────────
    home_adj = 0.0
    away_adj = 0.0
    adjustments_applied = []

    # Injury/suspension adjustments
    home_inj_adj = injury_adjustment(home_injuries or [])
    away_inj_adj = injury_adjustment(away_injuries or [])
    if home_inj_adj != 0.0:
        home_adj += home_inj_adj
        adjustments_applied.append(
            f"{home_team_name}: {len(home_injuries)} injury/suspension(s) ({home_inj_adj:+.0%})"
        )
    if away_inj_adj != 0.0:
        away_adj += away_inj_adj
        adjustments_applied.append(
            f"{away_team_name}: {len(away_injuries)} injury/suspension(s) ({away_inj_adj:+.0%})"
        )

    # FIX #4: xPts regression adjustment DISABLED — untested heuristic removed.
    # The −xpts_diff × 0.008 adjustment assumed all teams overperforming xPts are
    # "lucky" and due for regression. In practice, teams often outperform xPts for
    # structural reasons (elite finisher, defensive system, set-piece threat) and
    # the adjustment was just as likely to hurt as help. An independent backtest
    # on a temporal holdout is required before re-enabling. Until that validation
    # exists, applying it risks introducing systematic misdirection on ~30-40% of
    # matches. FBref data is still fetched (used for display on the frontend)
    # but the probability adjustment is suppressed.

    # Apply adjustments if any
    if home_adj != 0.0 or away_adj != 0.0:
        h, d, a = _apply_prob_adjustments(
            result["home_win_prob"], result["draw_prob"], result["away_win_prob"],
            home_adj, away_adj,
        )
        result["home_win_prob"] = round(h, 4)
        result["draw_prob"]     = round(d, 4)
        result["away_win_prob"] = round(a, 4)

        # Update predicted outcome after adjustment
        probs = {"HOME": h, "DRAW": d, "AWAY": a}
        result["predicted_outcome"] = max(probs, key=probs.get)

    # Attach enrichment metadata to result
    result["adjustments"]  = adjustments_applied
    result["home_injuries"] = home_injuries or []
    result["away_injuries"] = away_injuries or []
    result["home_fbref"]   = home_fbref or {}
    result["away_fbref"]   = away_fbref or {}
    result["pinnacle_odds"] = pinnacle_odds

    # ── CLV logging ───────────────────────────────────────────────────────────
    if home_team_name and away_team_name:
        # FIX #21: use the API's stable numeric match ID for CLV tracking instead
        # of the name-abbreviation slug. The old slug (e.g. "PL-2024-10-05-MANCHE-LIVE")
        # was non-unique for same-day derbies and silently collided in the ledger,
        # corrupting CLV measurements. Fall back to the slug only when no API ID.
        match_id = str(api_match_id) if api_match_id else _make_match_id(
            competition_code, date_str, home_team_name, away_team_name
        )
        # FIX #5: log over25 CLV now that Pinnacle totals odds are available.
        # btts / over35 remain logged without implied_prob (no Pinnacle line available).
        for market, prob_key in [
            ("home",   "home_win_prob"),
            ("draw",   "draw_prob"),
            ("away",   "away_win_prob"),
            ("over25", "over25_prob"),
        ]:
            try:
                log_prediction(
                    match_id=match_id,
                    match_date=date_str,
                    home_team=home_team_name,
                    away_team=away_team_name,
                    competition=_competition_name(competition_code),
                    market=market,
                    model_prob=result[prob_key],
                    opening_odds=bookmaker_odds,
                    pinnacle_opening_odds=pinnacle_odds,
                )
            except Exception:
                pass  # never block prediction due to CLV logging failure

    # ── Ledger ────────────────────────────────────────────────────────────────
    if save_to_ledger and home_team_name and away_team_name:
        league_name = _competition_name(competition_code)
        # FIX #21 (ledger path): same stable ID as CLV logging above
        match_id = str(api_match_id) if api_match_id else _make_match_id(
            competition_code, date_str, home_team_name, away_team_name
        )

        key_factors = _summarise_factors(
            home_team_name, away_team_name, result,
            home_xg, away_xg, h2h_matches, adjustments_applied,
        )

        ledger_entry = build_ledger_entry(
            match_id=match_id,
            api_match_id=api_match_id,
            date=date_str,
            league=league_name,
            home=home_team_name,
            away=away_team_name,
            prediction={
                "result":            result["predicted_outcome"],
                "confidence":        result["confidence"],
                "home_prob":         result["home_win_prob"],
                "draw_prob":         result["draw_prob"],
                "away_prob":         result["away_win_prob"],
                "over_2.5_prob":     result["over25_prob"],
                "btts_prob":         result["btts_prob"],
                "over_2.5_predicted": result["over25_predicted"],
                "btts_predicted":    result["btts_predicted"],
                "stars":             result["stars"],
                "dc_available":      result.get("dc_available", False),
            },
            factors_used=_factors_used(home_xg, h2h_matches, home_injuries, away_injuries),
            key_factors=key_factors,
        )
        append_prediction(ledger_entry)

    return result


def _factors_used(home_xg: float, h2h_matches: list, home_inj: list, away_inj: list) -> list[str]:
    factors = ["form", "goals", "standings", "home_advantage"]
    if h2h_matches:
        factors.append("h2h")
    if home_xg > 0:
        factors.append("xg")
    if home_inj or away_inj:
        factors.append("injuries")
    return factors


def _summarise_factors(
    home: str,
    away: str,
    result: dict,
    home_xg: float,
    away_xg: float,
    h2h_matches: list,
    adjustments: list[str],
) -> str:
    outcome_labels = {"HOME": f"{home} win", "DRAW": "draw", "AWAY": f"{away} win"}
    outcome = outcome_labels.get(result["predicted_outcome"], result["predicted_outcome"])
    parts = [
        f"Model predicts {outcome} ({result['confidence']:.0%} confidence, {result['stars']}★).",
    ]
    if home_xg > 0:
        parts.append(f"xG: {home} {home_xg:.2f} | {away} {away_xg:.2f}.")
    if result["over25_prob"] >= 0.5:
        parts.append(f"Over 2.5 likely ({result['over25_prob']:.0%}).")
    if result["btts_prob"] >= 0.5:
        parts.append(f"BTTS likely ({result['btts_prob']:.0%}).")
    if result["value_bets"]:
        parts.append(f"Value: {', '.join(result['value_bets'][:2])}.")
    if adjustments:
        parts.append(f"Adjustments: {'; '.join(adjustments[:2])}.")
    return " ".join(parts)


async def predict_upcoming_batch(
    competition_code: str = "PL",
    days_ahead: int = 1,
    save_to_ledger: bool = True,
) -> list[dict]:
    """
    Predict all upcoming matches for a competition.
    Returns list of {match_info, prediction} dicts.
    """
    from app.services.football_api import get_upcoming_matches

    matches = await _safe(get_upcoming_matches(competition_code, days_ahead), [])
    results = []

    for i, match in enumerate(matches):
        home_id = match.get("homeTeam", {}).get("id")
        away_id = match.get("awayTeam", {}).get("id")
        if not home_id or not away_id:
            continue

        home_name = (
            match.get("homeTeam", {}).get("shortName")
            or match.get("homeTeam", {}).get("name", "")
        )
        away_name = (
            match.get("awayTeam", {}).get("shortName")
            or match.get("awayTeam", {}).get("name", "")
        )
        match_date = (match.get("utcDate") or "")[:10]

        try:
            pred = await predict_match(
                home_team_id=home_id,
                away_team_id=away_id,
                competition_code=competition_code,
                api_match_id=match.get("id"),
                home_team_name=home_name,
                away_team_name=away_name,
                match_date=match_date,
                save_to_ledger=save_to_ledger,
            )
        except Exception as e:
            pred = {"error": str(e)}

        results.append({
            "api_match_id": match.get("id"),
            "match_date":   match.get("utcDate"),
            "home_team":    home_name,
            "away_team":    away_name,
            "competition":  _competition_name(competition_code),
            "prediction":   pred,
        })

        if (i + 1) % 3 == 0:
            await asyncio.sleep(7)

    return results
