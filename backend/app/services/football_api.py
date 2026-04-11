"""
Client for football-data.org free API.
Free tier: 10 req/min. Competitions: PL, CL, BL1, PD, SA, FL1, PPL, ELC, EC, WC, DED, PPL, BSA.
Sign up at https://www.football-data.org/client/register

Rate limiting:
  Free tier allows 10 req/min (= 1 req / 6s). A module-level semaphore
  (1 slot) plus a minimum inter-request delay of MIN_INTERVAL seconds ensures
  concurrent callers can't exceed this limit even during batch operations.
"""
import asyncio
import time
import httpx
from datetime import datetime, timedelta
from typing import Optional
from app.config import settings

BASE_URL = "https://api.football-data.org/v4"

# Rate limiting: 10 req/min free tier → 1 per 6s; we use 7s for safety
_semaphore = asyncio.Semaphore(1)
_last_request_time: float = 0.0
MIN_INTERVAL: float = 7.0  # seconds between requests

# All priority leagues — codes mapped to display names
SUPPORTED_COMPETITIONS = {
    "PL":  "Premier League",
    "ELC": "Championship",
    "CL":  "Champions League",
    "EL":  "Europa League",
    "ECL": "Conference League",
    "PD":  "La Liga",
    "SA":  "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "PPL2": "Scottish Premiership",  # Note: use BSA or PPL depending on API support
}

# Competitions available on football-data.org free tier
FDORG_COMPETITIONS = {"PL", "ELC", "CL", "PD", "SA", "BL1", "FL1", "DED", "PPL"}


def _headers() -> dict:
    return {"X-Auth-Token": settings.football_data_api_key}


async def _get(url: str, params: Optional[dict] = None) -> dict:
    """
    Rate-limited GET request — enforces at most 1 request per MIN_INTERVAL seconds
    globally across all concurrent callers via a semaphore.
    """
    global _last_request_time
    async with _semaphore:
        now = time.monotonic()
        wait = MIN_INTERVAL - (now - _last_request_time)
        if wait > 0:
            await asyncio.sleep(wait)
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=_headers(), params=params)
            resp.raise_for_status()
            _last_request_time = time.monotonic()
            return resp.json()


async def get_upcoming_matches(competition_code: str = "PL", days_ahead: int = 7) -> list[dict]:
    date_from = datetime.utcnow().strftime("%Y-%m-%d")
    date_to = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    data = await _get(
        f"{BASE_URL}/competitions/{competition_code}/matches",
        {"status": "SCHEDULED", "dateFrom": date_from, "dateTo": date_to},
    )
    return data.get("matches", [])


async def get_today_matches(competition_code: str = "PL") -> list[dict]:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    data = await _get(
        f"{BASE_URL}/competitions/{competition_code}/matches",
        {"status": "SCHEDULED", "dateFrom": today, "dateTo": today},
    )
    return data.get("matches", [])


async def get_tomorrow_matches(competition_code: str = "PL") -> list[dict]:
    tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    data = await _get(
        f"{BASE_URL}/competitions/{competition_code}/matches",
        {"status": "SCHEDULED", "dateFrom": tomorrow, "dateTo": tomorrow},
    )
    return data.get("matches", [])


async def get_finished_matches(competition_code: str = "PL", limit: int = 100) -> list[dict]:
    # football-data.org v4 does not accept a 'limit' param — returns full season, we slice
    data = await _get(
        f"{BASE_URL}/competitions/{competition_code}/matches",
        {"status": "FINISHED"},
    )
    matches = data.get("matches", [])
    return matches[-limit:] if limit else matches


async def get_live_matches(competition_code: str = "PL") -> list[dict]:
    data = await _get(
        f"{BASE_URL}/competitions/{competition_code}/matches",
        {"status": "IN_PLAY,PAUSED"},
    )
    return data.get("matches", [])


async def get_standings(competition_code: str = "PL") -> list[dict]:
    data = await _get(f"{BASE_URL}/competitions/{competition_code}/standings")
    for s in data.get("standings", []):
        if s.get("type") == "TOTAL":
            return s.get("table", [])
    return []


async def get_team_matches(team_id: int, limit: int = 20, status: str = "FINISHED") -> list[dict]:
    data = await _get(
        f"{BASE_URL}/teams/{team_id}/matches",
        {"status": status, "limit": limit},
    )
    return data.get("matches", [])


async def get_team_home_matches(team_id: int, limit: int = 15) -> list[dict]:
    """Finished home matches only."""
    matches = await get_team_matches(team_id, limit=40)
    return [m for m in matches if m.get("homeTeam", {}).get("id") == team_id][:limit]


async def get_team_away_matches(team_id: int, limit: int = 15) -> list[dict]:
    """Finished away matches only."""
    matches = await get_team_matches(team_id, limit=40)
    return [m for m in matches if m.get("awayTeam", {}).get("id") == team_id][:limit]


async def get_h2h(match_id: int) -> list[dict]:
    """Head-to-head using the match's own H2H endpoint."""
    data = await _get(f"{BASE_URL}/matches/{match_id}/head2head", {"limit": 10})
    return data.get("matches", [])


async def get_team_info(team_id: int) -> dict:
    return await _get(f"{BASE_URL}/teams/{team_id}")


async def get_all_today_matches() -> list[dict]:
    """Pull today's matches across all supported FDORG competitions."""
    results = []
    for code in FDORG_COMPETITIONS:
        try:
            matches = await get_today_matches(code)
            for m in matches:
                m["_competition_code"] = code
                m["_competition_name"] = SUPPORTED_COMPETITIONS.get(code, code)
            results.extend(matches)
        except Exception:
            pass
    return results


async def get_all_tomorrow_matches() -> list[dict]:
    results = []
    for code in FDORG_COMPETITIONS:
        try:
            matches = await get_tomorrow_matches(code)
            for m in matches:
                m["_competition_code"] = code
                m["_competition_name"] = SUPPORTED_COMPETITIONS.get(code, code)
            results.extend(matches)
        except Exception:
            pass
    return results


async def find_team_by_name(name: str) -> Optional[dict]:
    """Search for a team by name (uses the teams endpoint)."""
    try:
        data = await _get(f"{BASE_URL}/teams", {"name": name, "limit": 5})
        teams = data.get("teams", [])
        return teams[0] if teams else None
    except Exception:
        return None
