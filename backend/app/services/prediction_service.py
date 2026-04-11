"""
Orchestrates data fetching, feature engineering, prediction, and ledger logging
for a single match or a batch of upcoming matches.
"""
import asyncio
from datetime import datetime
from typing import Optional

from app.services.football_api import (
    get_team_matches,
    get_standings,
    get_h2h,
    SUPPORTED_COMPETITIONS,
)
from app.services.scraper import fetch_understat_team_xg
from app.services.evaluator import append_prediction, build_ledger_entry
from ml.features import build_feature_vector
from ml.predict import predict


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
    Returns a comprehensive prediction dict (see ml.predict.predict).
    """
    # Fetch team data in parallel
    home_matches, away_matches, standings_table = await asyncio.gather(
        _safe(get_team_matches(home_team_id, limit=25)),
        _safe(get_team_matches(away_team_id, limit=25)),
        _safe(get_standings(competition_code)),
    )

    # H2H — only if we have a match ID
    h2h_matches = []
    if api_match_id:
        h2h_matches = await _safe(get_h2h(api_match_id), [])

    # xG enrichment (best effort — doesn't block if Understat fails)
    home_xg_data, away_xg_data = await asyncio.gather(
        _safe(fetch_understat_team_xg(home_team_name, competition_code), {}),
        _safe(fetch_understat_team_xg(away_team_name, competition_code), {}),
    )
    home_xg = home_xg_data.get("last5_xg_for", 0.0) if home_xg_data else 0.0
    away_xg = away_xg_data.get("last5_xg_for", 0.0) if away_xg_data else 0.0

    standing_map = {row["team"]["id"]: row for row in (standings_table or [])}

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
    )

    result = predict(vec, bookmaker_odds=bookmaker_odds)

    # Optionally persist to ledger
    if save_to_ledger and home_team_name and away_team_name:
        league_name = _competition_name(competition_code)
        date_str = match_date or datetime.utcnow().strftime("%Y-%m-%d")
        match_id = _make_match_id(competition_code, date_str, home_team_name, away_team_name)

        key_factors = _summarise_factors(
            home_team_name, away_team_name, result,
            home_xg, away_xg, h2h_matches,
        )

        ledger_entry = build_ledger_entry(
            match_id=match_id,
            api_match_id=api_match_id,
            date=date_str,
            league=league_name,
            home=home_team_name,
            away=away_team_name,
            prediction={
                "result": result["predicted_outcome"],
                "confidence": result["confidence"],
                "over_2.5_prob": result["over25_prob"],
                "btts_prob": result["btts_prob"],
                "over_2.5_predicted": result["over25_predicted"],
                "btts_predicted": result["btts_predicted"],
                "stars": result["stars"],
            },
            factors_used=_factors_used(home_xg, h2h_matches),
            key_factors=key_factors,
        )
        append_prediction(ledger_entry)

    return result


def _factors_used(home_xg: float, h2h_matches: list) -> list[str]:
    factors = ["form", "goals", "standings", "home_advantage"]
    if h2h_matches:
        factors.append("h2h")
    if home_xg > 0:
        factors.append("xg")
    return factors


def _summarise_factors(
    home: str,
    away: str,
    result: dict,
    home_xg: float,
    away_xg: float,
    h2h_matches: list,
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

        # Rate limit — pause between matches
        if (i + 1) % 3 == 0:
            await asyncio.sleep(7)

    return results
