import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

# Importing prediction logic and API fetchers
from app.services.prediction_service import predict_match, predict_upcoming_batch
from app.services.football_api import (
    get_upcoming_matches, get_all_today_matches,
    SUPPORTED_COMPETITIONS, FDORG_COMPETITIONS,
)

# Initialize the router with a prefix and tags for Swagger documentation
router = APIRouter(prefix="/predictions", tags=["predictions"])

# --- In-memory cache for today's predictions ---
# This avoids hitting the ML model and external APIs on every page refresh.
# Structure: { date_str: {"status": "computing"|"ready", "predictions": [...], "total": int, "done": int} }
_today_cache: dict = {}
_preload_running: bool = False


def _today_str() -> str:
    """Returns today's date as a YYYY-MM-DD string."""
    return datetime.utcnow().strftime("%Y-%m-%d")


async def preload_today_predictions() -> None:
    """
    BACKGROUND TASK: This is the core logic that prepares today's tips.
    1. Fetches all matches for today across all supported leagues.
    2. Runs the ML model for each match.
    3. Progressively updates the in-memory cache so the frontend stays updated.
    """
    global _preload_running
    if _preload_running:
        return  # Prevent multiple preloads from running at the same time
    _preload_running = True

    # Check if we need to refresh the standings/data cache before predicting
    try:
        from app.services.api_cache import any_stale
        from app.api.admin import _refresh_state, _run_cache_refresh, _refresh_lock
        standings_keys = [f"standings_{c}" for c in FDORG_COMPETITIONS]
        if any_stale(standings_keys):
            async with _refresh_lock:
                if _refresh_state["status"] != "running":
                    print("[preload] API cache is stale — triggering background refresh")
                    asyncio.create_task(_run_cache_refresh())
    except Exception:
        pass  # Never block predictions due to minor cache errors

    date_str = _today_str()
    # Initialize the cache entry for today
    _today_cache[date_str] = {"status": "computing", "predictions": [], "done": 0, "total": 0}

    try:
        # Fetch today's schedule
        matches = await get_all_today_matches()
        valid = [
            m for m in matches
            if m.get("homeTeam", {}).get("id") and m.get("awayTeam", {}).get("id")
        ]
        _today_cache[date_str]["total"] = len(valid)
        print(f"[preload] {len(valid)} matches to predict for {date_str}")

        # Loop through matches and run predictions one by one
        for m in valid:
            home_id   = m["homeTeam"]["id"]
            away_id   = m["awayTeam"]["id"]
            home_name = m["homeTeam"].get("shortName") or m["homeTeam"].get("name", "")
            away_name = m["awayTeam"].get("shortName") or m["awayTeam"].get("name", "")
            comp_code = m.get("_competition_code", "PL")
            match_date = (m.get("utcDate") or "")[:10]

            try:
                # RUN THE ML MODEL
                pred = await predict_match(
                    home_team_id=home_id,
                    away_team_id=away_id,
                    competition_code=comp_code,
                    api_match_id=m.get("id"),
                    home_team_name=home_name,
                    away_team_name=away_name,
                    match_date=match_date,
                )
                
                # Append result to the cache
                _today_cache[date_str]["predictions"].append({
                    "api_match_id":     m.get("id"),
                    "match_date":       m.get("utcDate"),
                    "home_team":        home_name,
                    "away_team":        away_name,
                    "home_team_crest":  m["homeTeam"].get("crest"),
                    "away_team_crest":  m["awayTeam"].get("crest"),
                    "competition":      m.get("_competition_name", comp_code),
                    "competition_code": comp_code,
                    "prediction":       pred,
                })
            except Exception as e:
                print(f"[preload] prediction failed for {home_name} vs {away_name}: {e}")

            # Update progress counter
            _today_cache[date_str]["done"] += 1

        _today_cache[date_str]["status"] = "ready"
        print(f"[preload] done — {_today_cache[date_str]['done']} predictions cached")

    except Exception as e:
        print(f"[preload] failed: {e}")
        if date_str in _today_cache:
            _today_cache[date_str]["status"] = "error"
    finally:
        _preload_running = False


# --- Request Models (Pydantic) ---

class PredictRequest(BaseModel):
    """Data required to request a prediction for a single match."""
    home_team_id: int
    away_team_id: int
    competition_code: str = "PL"
    home_team_name: str = ""
    away_team_name: str = ""
    match_date: str = ""
    save_to_ledger: bool = False
    bookmaker_odds: Optional[dict] = None


# --- API Endpoints ---

@router.post("/predict")
async def predict_single(body: PredictRequest):
    """Allows manual triggering of a prediction for a specific game."""
    if body.competition_code not in FDORG_COMPETITIONS:
        raise HTTPException(400, f"Unsupported competition. Choose from: {list(FDORG_COMPETITIONS)}")
    try:
        result = await predict_match(
            home_team_id=body.home_team_id,
            away_team_id=body.away_team_id,
            competition_code=body.competition_code,
            home_team_name=body.home_team_name,
            away_team_name=body.away_team_name,
            match_date=body.match_date,
            save_to_ledger=body.save_to_ledger,
            bookmaker_odds=body.bookmaker_odds,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@router.get("/today")
async def today_predictions():
    """
    MAIN DASHBOARD ENDPOINT: Returns all cached predictions for today.
    Status will be 'computing' if the background task is still running.
    """
    date_str = _today_str()
    cached = _today_cache.get(date_str)
    if not cached:
        return {"status": "idle", "predictions": [], "done": 0, "total": 0, "date": date_str}
    return {**cached, "date": date_str}


@router.get("/clv")
async def clv_stats(days: int = 30):
    """
    Returns 'Closing Line Value' performance stats.
    This shows if the AI is consistently beating the bookmaker's closing odds.
    """
    from app.services.clv_tracker import get_clv_stats
    return get_clv_stats(days=days)


@router.post("/preload")
async def trigger_preload():
    """Manually force the background prediction task to start."""
    date_str = _today_str()
    cached = _today_cache.get(date_str)
    if cached and cached["status"] in ("computing", "ready"):
        return {"message": f"Already {cached['status']}", "status": cached["status"]}
    asyncio.create_task(preload_today_predictions())
    return {"message": "Preload started", "status": "computing"}


@router.get("/upcoming")
async def predict_upcoming(
    competition: str = Query("PL"),
    days_ahead: int = Query(1, ge=1, le=14),
    save_to_ledger: bool = Query(False),
):
    """Returns predictions for matches happening in the next few days."""
    if competition not in FDORG_COMPETITIONS:
        raise HTTPException(400, "Unsupported competition.")
    try:
        results = await predict_upcoming_batch(competition, days_ahead, save_to_ledger)
        return {"competition": competition, "predictions": results}
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch predictions: {e}")

