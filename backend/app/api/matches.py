from fastapi import APIRouter, HTTPException, Query
from app.services.football_api import (
    get_upcoming_matches,
    get_finished_matches,
    get_standings,
    get_all_today_matches,
    get_team_matches,
    find_team_by_name,
    SUPPORTED_COMPETITIONS,
    FDORG_COMPETITIONS,
)
from app.services.rapidapi_football import (
    get_sofascore_team_xg,
    get_free_api_live_scores,
    get_sportapi_live_scores,
)

router = APIRouter(prefix="/matches", tags=["matches"])


@router.get("/upcoming")
async def upcoming_matches(
    competition: str = Query("PL", description="Competition code e.g. PL, PD, BL1"),
    days_ahead: int = Query(7, ge=1, le=30),
):
    if competition not in FDORG_COMPETITIONS:
        raise HTTPException(400, f"Unsupported competition. Choose from: {list(FDORG_COMPETITIONS)}")
    try:
        matches = await get_upcoming_matches(competition, days_ahead)
        return {"competition": competition, "matches": matches}
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch matches: {e}")


@router.get("/today")
async def today_matches():
    """Today's matches across all tracked leagues."""
    try:
        matches = await get_all_today_matches()
        return {"date": "today", "count": len(matches), "matches": matches}
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch today's matches: {e}")


@router.get("/results")
async def recent_results(
    competition: str = Query("PL"),
    limit: int = Query(20, ge=1, le=100),
):
    if competition not in FDORG_COMPETITIONS:
        raise HTTPException(400, "Unsupported competition.")
    try:
        matches = await get_finished_matches(competition, limit)
        return {"competition": competition, "matches": matches}
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch results: {e}")


@router.get("/standings")
async def standings(competition: str = Query("PL")):
    if competition not in FDORG_COMPETITIONS:
        raise HTTPException(400, "Unsupported competition.")
    try:
        table = await get_standings(competition)
        return {"competition": competition, "table": table}
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch standings: {e}")


@router.get("/competitions")
async def list_competitions():
    return {
        "all": SUPPORTED_COMPETITIONS,
        "available": {k: v for k, v in SUPPORTED_COMPETITIONS.items() if k in FDORG_COMPETITIONS},
    }


@router.get("/search")
async def search_team(q: str = Query(..., min_length=2, description="Team name to search")):
    """Find a team by name (fuzzy match). Returns the best matching team object."""
    team = await find_team_by_name(q)
    if not team:
        raise HTTPException(404, f"No team found matching '{q}'")
    return {"team": team}


@router.get("/team/{team_id}/form")
async def team_form(
    team_id: int,
    limit: int = Query(10, ge=1, le=30, description="Number of recent matches"),
):
    """Recent finished matches for a team (for form guide)."""
    try:
        matches = await get_team_matches(team_id, limit=limit, status="FINISHED")
        return {"team_id": team_id, "matches": matches}
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch team form: {e}")


@router.get("/team/{team_id}/upcoming")
async def team_upcoming(
    team_id: int,
    limit: int = Query(5, ge=1, le=20),
):
    """Upcoming scheduled matches for a team."""
    try:
        matches = await get_team_matches(team_id, limit=limit, status="SCHEDULED")
        return {"team_id": team_id, "matches": matches}
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch upcoming fixtures: {e}")


@router.get("/live")
async def live_scores():
    """
    Currently in-play matches from RapidAPI (Sofascore / Free API fallback).
    Cached for 90 seconds. Returns [] gracefully when no key is configured.
    """
    try:
        scores = await get_sportapi_live_scores()
        return {"count": len(scores), "matches": scores}
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch live scores: {e}")


@router.get("/team/xg")
async def team_xg(
    name: str = Query(..., description="Team name"),
    competition: str = Query("PL"),
):
    """
    Return a team's rolling xG stats from Sofascore.
    Result: {last5_xg_for, last5_xg_against}
    """
    try:
        xg = await get_sofascore_team_xg(name, competition)
        if not xg:
            raise HTTPException(404, f"No xG data found for '{name}'")
        return {"team": name, "competition": competition, **xg}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"xG fetch failed: {e}")
