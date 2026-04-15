"""
Client for football-data.org free API.
Free tier: 10 req/min. Competitions: PL, CL, BL1, PD, SA, FL1, PPL, ELC, EC, WC, DED, PPL, BSA.
Sign up at https://www.football-data.org/client/register
"""
import asyncio
import httpx
from datetime import datetime, timedelta
from typing import Optional
from app.config import settings

BASE_URL = "https://api.football-data.org/v4"

# Rate limiting — free tier: 10 req/min → enforce min 6.5s gap between requests
_rate_lock = asyncio.Lock()
_last_request_time: float = 0.0

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
    """Rate-limited GET — enforces 6.5s minimum gap to stay within 10 req/min free tier."""
    global _last_request_time
    async with _rate_lock:
        now = asyncio.get_event_loop().time()
        wait = max(0.0, 6.5 - (now - _last_request_time))
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=_headers(), params=params)
            if resp.status_code == 429:
                await asyncio.sleep(60)
                resp = await client.get(url, headers=_headers(), params=params)
            resp.raise_for_status()
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
    from app.services.api_cache import get as _cache_get, set as _cache_set
    key = f"finished_{competition_code}"
    cached = _cache_get(key)
    if cached is not None:
        return cached[-limit:] if limit else cached
    data = await _get(
        f"{BASE_URL}/competitions/{competition_code}/matches",
        {"status": "FINISHED"},
    )
    matches = data.get("matches", [])
    _cache_set(key, matches)
    return matches[-limit:] if limit else matches


async def get_live_matches(competition_code: str = "PL") -> list[dict]:
    data = await _get(
        f"{BASE_URL}/competitions/{competition_code}/matches",
        {"status": "IN_PLAY,PAUSED"},
    )
    return data.get("matches", [])


async def get_standings(competition_code: str = "PL") -> list[dict]:
    from app.services.api_cache import get as _cache_get, set as _cache_set
    key = f"standings_{competition_code}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    data = await _get(f"{BASE_URL}/competitions/{competition_code}/standings")
    for s in data.get("standings", []):
        if s.get("type") == "TOTAL":
            table = s.get("table", [])
            _cache_set(key, table)
            return table
    return []


async def get_team_matches(team_id: int, limit: int = 20, status: str = "FINISHED") -> list[dict]:
    # Only cache FINISHED history — SCHEDULED/IN_PLAY change too fast
    if status == "FINISHED":
        from app.services.api_cache import get as _cache_get, set as _cache_set
        key = f"team_{team_id}"
        cached = _cache_get(key)
        if cached is not None:
            return cached[:limit] if limit else cached
        data = await _get(
            f"{BASE_URL}/teams/{team_id}/matches",
            {"status": status, "limit": limit},
        )
        matches = data.get("matches", [])
        _cache_set(key, matches)
        return matches[:limit] if limit else matches
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
            await asyncio.sleep(6)  # stay within 10 req/min
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
            await asyncio.sleep(6)
        except Exception:
            pass
    return results


# ── Team lookup cache ─────────────────────────────────────────────────────────
# The free-tier /v4/teams?name=... endpoint ignores the name filter and returns
# all teams (sorted so "1. FC Köln" always comes first). Instead we fetch each
# competition's team list once at startup and search locally.

_team_cache: dict[str, dict] = {}   # lowercase name/shortName → team dict
_team_cache_lock = asyncio.Lock()
_team_cache_ready = False


async def build_team_cache() -> None:
    """
    Fetch every competition's team list and build a local lookup cache.
    Uses its own httpx session with a 2s delay — bypasses the prediction
    rate limiter so cache build never queues behind live prediction requests.
    Called once at startup via the scheduler.
    """
    global _team_cache, _team_cache_ready
    async with _team_cache_lock:
        if _team_cache_ready:
            return
        added = 0
        async with httpx.AsyncClient(timeout=20) as client:
            for code in FDORG_COMPETITIONS:
                try:
                    resp = await client.get(
                        f"{BASE_URL}/competitions/{code}/teams",
                        headers=_headers(),
                    )
                    resp.raise_for_status()
                    for team in resp.json().get("teams", []):
                        for field in ("name", "shortName"):
                            val = (team.get(field) or "").strip()
                            if val:
                                _team_cache[val.lower()] = team
                                added += 1
                except Exception:
                    pass
                await asyncio.sleep(2)   # gentle pacing, separate from rate limiter
        _team_cache_ready = True
        print(f"[football_api] Team cache built: {len(_team_cache)} entries from {added} names")


async def find_team_by_name(name: str) -> Optional[dict]:
    """
    Find a team by name using the local cache (built at startup).
    Waits up to 45s for the cache if it isn't ready yet.
    """
    from difflib import get_close_matches

    # Wait for cache to be ready (built 15s after startup)
    if not _team_cache_ready:
        for _ in range(45):
            await asyncio.sleep(1)
            if _team_cache_ready:
                break

    name_lower = name.lower()
    # Use length >= 3 so short words like "Man", "FC", "AC" are included
    words = [w for w in name_lower.split() if len(w) >= 3]

    if _team_cache:
        # 1. Exact match
        if name_lower in _team_cache:
            return _team_cache[name_lower]

        # 2. All significant words present in the key
        if words:
            for key, team in _team_cache.items():
                if all(w in key for w in words):
                    return team

        # 3. Fuzzy match (difflib)
        candidates = list(_team_cache.keys())
        close = get_close_matches(name_lower, candidates, n=1, cutoff=0.65)
        if close:
            return _team_cache[close[0]]

    return None
