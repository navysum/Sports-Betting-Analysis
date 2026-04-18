"""
RapidAPI sports data clients — three free-tier subscriptions sharing one key.

1. Sofascore  (Api Dojo)              host: sofascore.p.rapidapi.com
   → real xG per match, match statistics, player ratings

2. Free API Live Football Data        host: free-api-live-football-data.p.rapidapi.com
   (Smart API)
   → fixtures by date, live scores, standings — 2100+ leagues

3. SportAPI  (rapidsportapi)          host: rapidsportapi.p.rapidapi.com
   → fast fixtures + results backup (141 ms average)

All functions return [] or {} on error so the prediction pipeline is never
blocked by a third-party outage. Results are cached to disk to stay well
within free-tier request budgets.

Set RAPIDAPI_KEY in your .env / Render environment variables.
"""

import asyncio
import json
import os
import time
from typing import Optional

import httpx

from app.config import settings

# ── API hosts ─────────────────────────────────────────────────────────────────
SOFASCORE_HOST     = "sofascore.p.rapidapi.com"
FREE_FOOTBALL_HOST = "free-api-live-football-data.p.rapidapi.com"
SPORT_API_HOST     = "rapidsportapi.p.rapidapi.com"

# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
)
_CACHE_PATH = os.path.join(_CACHE_DIR, "rapidapi_cache.json")

_XG_TTL      = 6 * 3600   # xG rolling averages: 6 h
_FIXTURES_TTL = 3600       # fixtures: 1 h
_LIVE_TTL    = 90          # live scores: 90 s
_TEAM_ID_TTL = 30 * 86400  # Sofascore team IDs never change: 30 days

_mem: dict = {}


def _cache_get(key: str, ttl: float) -> Optional[object]:
    if key in _mem:
        entry = _mem[key]
        if time.time() - entry["ts"] < ttl:
            return entry["data"]
    try:
        with open(_CACHE_PATH) as f:
            disk = json.load(f)
        entry = disk.get(key)
        if entry and time.time() - entry.get("ts", 0) < ttl:
            _mem[key] = entry
            return entry["data"]
    except Exception:
        pass
    return None


def _cache_set(key: str, data: object) -> None:
    entry = {"data": data, "ts": time.time()}
    _mem[key] = entry
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        try:
            with open(_CACHE_PATH) as f:
                disk = json.load(f)
        except Exception:
            disk = {}
        disk[key] = entry
        with open(_CACHE_PATH, "w") as f:
            json.dump(disk, f)
    except Exception:
        pass


# ── Shared rate-limited HTTP client ──────────────────────────────────────────
_rate_lock             = asyncio.Lock()
_last_call: dict[str, float] = {}   # host → monotonic time of last call
_MIN_GAP   = 2.0                    # seconds between calls to the same host


def _headers(host: str) -> dict:
    return {
        "X-RapidAPI-Key":  settings.rapidapi_key,
        "X-RapidAPI-Host": host,
    }


async def _get(
    host: str,
    path: str,
    params: Optional[dict] = None,
    min_gap: float = _MIN_GAP,
) -> Optional[dict]:
    """Rate-limited GET for any RapidAPI host. Returns None on any error."""
    if not settings.rapidapi_key:
        return None

    async with _rate_lock:
        now  = time.monotonic()
        last = _last_call.get(host, 0.0)
        wait = max(0.0, min_gap - (now - last))
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call[host] = time.monotonic()

    url = f"https://{host}{path}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=_headers(host), params=params or {})
            if resp.status_code == 429:
                print(f"[rapidapi] 429 rate-limit on {host} — waiting 60 s")
                await asyncio.sleep(60)
                resp = await client.get(url, headers=_headers(host), params=params or {})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"[rapidapi] {host}{path} → {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Sofascore — xG and match statistics
# ═══════════════════════════════════════════════════════════════════════════════

async def _sofascore_find_team_id(name: str) -> Optional[int]:
    """
    Search Sofascore for a football team by name.
    Returns the Sofascore team ID or None if not found.
    IDs are cached for 30 days — they never change.
    """
    cache_key = f"ss_tid_{name.lower().replace(' ', '_')}"
    cached = _cache_get(cache_key, _TEAM_ID_TTL)
    if cached is not None:
        return int(cached)

    data = await _get(SOFASCORE_HOST, "/api/v1/search/all", {"q": name, "page": "0"})
    if not data:
        return None

    name_lower = name.lower()
    best_id: Optional[int] = None
    for result in data.get("results", []):
        if result.get("type") != "team":
            continue
        entity = result.get("entity", {})
        # sport id 1 = football
        if entity.get("sport", {}).get("id") != 1:
            continue
        entity_name = (entity.get("name") or "").lower()
        if name_lower in entity_name or entity_name in name_lower:
            best_id = entity.get("id")
            break
        if best_id is None:
            best_id = entity.get("id")   # take first football team as fallback

    if best_id:
        _cache_set(cache_key, best_id)
    return best_id


async def _sofascore_last_events(team_id: int) -> list[dict]:
    """Return the most-recent Sofascore events for a team (last page = 0)."""
    data = await _get(SOFASCORE_HOST, f"/api/v1/team/{team_id}/events/last/0")
    return (data or {}).get("events", [])


async def _sofascore_event_xg(
    event_id: int,
    is_home: bool,
) -> Optional[tuple[float, float]]:
    """
    Fetch xG (for, against) from a Sofascore event for one side.

    Returns (xg_for, xg_against) or None if xG data is unavailable.
    Finished-match statistics are cached for 30 days.
    """
    cache_key = f"ss_xg_{event_id}"
    cached = _cache_get(cache_key, _TEAM_ID_TTL)
    if cached is not None:
        h_xg, a_xg = cached
        return (h_xg, a_xg) if is_home else (a_xg, h_xg)

    data = await _get(SOFASCORE_HOST, f"/api/v1/event/{event_id}/statistics")
    if not data:
        return None

    for block in data.get("statistics", []):
        if block.get("period") != "ALL":
            continue
        for group in block.get("groups", []):
            for item in group.get("statisticsItems", []):
                name = (item.get("name") or "").lower()
                if "expected" in name and "goal" in name:
                    try:
                        h_xg = float(item.get("home") or 0)
                        a_xg = float(item.get("away") or 0)
                        _cache_set(cache_key, [h_xg, a_xg])
                        return (h_xg, a_xg) if is_home else (a_xg, h_xg)
                    except (TypeError, ValueError):
                        pass
    return None


async def get_sofascore_team_xg(
    team_name: str,
    competition_code: str = "PL",
) -> dict:
    """
    Return a team's rolling xG stats from Sofascore.

    Walks the team's last 10 finished matches, extracts xG from the event
    statistics endpoint, and returns the average over up to 5 matches.

    Return format (same as fetch_understat_team_xg):
        {"last5_xg_for": float, "last5_xg_against": float}

    Returns {} on any failure — callers fall back to Understat or the xG proxy.
    """
    cache_key = f"ss_xg_{team_name.lower().replace(' ', '_')}_{competition_code}"
    cached = _cache_get(cache_key, _XG_TTL)
    if cached is not None:
        return cached

    team_id = await _sofascore_find_team_id(team_name)
    if not team_id:
        print(f"[sofascore] team not found: {team_name}")
        return {}

    events = await _sofascore_last_events(team_id)
    if not events:
        return {}

    xg_for_list:     list[float] = []
    xg_against_list: list[float] = []

    for event in events:
        # Only use finished matches
        status = (event.get("status") or {}).get("type", "")
        if status not in ("finished",):
            continue

        event_id = event.get("id")
        if not event_id:
            continue

        home_id  = (event.get("homeTeam") or {}).get("id")
        is_home  = (home_id == team_id)

        xg_pair = await _sofascore_event_xg(event_id, is_home=is_home)
        if xg_pair is None:
            continue

        xg_for_list.append(xg_pair[0])
        xg_against_list.append(xg_pair[1])

        if len(xg_for_list) >= 5:
            break

    if not xg_for_list:
        print(f"[sofascore] no xG data found for {team_name}")
        return {}

    result = {
        "last5_xg_for":     round(sum(xg_for_list)     / len(xg_for_list),     3),
        "last5_xg_against": round(sum(xg_against_list) / len(xg_against_list), 3),
    }
    print(f"[sofascore] {team_name}: xG for={result['last5_xg_for']:.2f}  against={result['last5_xg_against']:.2f}")
    _cache_set(cache_key, result)
    return result


async def get_sofascore_match_stats(event_id: int) -> dict:
    """
    Return full match statistics for a Sofascore event ID.
    Useful for post-match enrichment (shots, possession, xG).
    """
    data = await _get(SOFASCORE_HOST, f"/api/v1/event/{event_id}/statistics")
    return data or {}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Free API Live Football Data — fixtures + live scores
# ═══════════════════════════════════════════════════════════════════════════════

def _unwrap_list(data: Optional[dict]) -> list:
    """Extract the match list from various response shapes."""
    if not data:
        return []
    for key in ("response", "matches", "data", "result", "fixtures"):
        val = data.get(key)
        if isinstance(val, list):
            return val
    return []


async def get_free_api_fixtures(date: str) -> list[dict]:
    """
    Fetch all football fixtures for a given date (YYYY-MM-DD).
    Covers 2100+ leagues — far broader than football-data.org free tier.
    """
    cache_key = f"free_fixtures_{date}"
    cached = _cache_get(cache_key, _FIXTURES_TTL)
    if cached is not None:
        return cached

    data = await _get(FREE_FOOTBALL_HOST, "/get-matches-by-date", {"date": date})
    matches = _unwrap_list(data)
    if matches:
        _cache_set(cache_key, matches)
    return matches


async def get_free_api_live_scores() -> list[dict]:
    """
    Fetch currently in-play football matches.
    Cached for 90 seconds — short enough to reflect score changes quickly.
    """
    cache_key = "free_live_scores"
    cached = _cache_get(cache_key, _LIVE_TTL)
    if cached is not None:
        return cached

    data = await _get(FREE_FOOTBALL_HOST, "/get-live-matches-score")
    matches = _unwrap_list(data)
    if matches:
        _cache_set(cache_key, matches)
    return matches


async def get_free_api_standings(league_id: str) -> list[dict]:
    """
    Fetch standings for a league from Free API Live Football Data.
    league_id: the API's internal league identifier string.
    """
    cache_key = f"free_standings_{league_id}"
    cached = _cache_get(cache_key, _FIXTURES_TTL * 6)  # 6 h — standings rarely change
    if cached is not None:
        return cached

    data = await _get(FREE_FOOTBALL_HOST, "/get-league-standings", {"leagueId": league_id})
    rows = _unwrap_list(data)
    if rows:
        _cache_set(cache_key, rows)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SportAPI — fast fixtures + results backup
# ═══════════════════════════════════════════════════════════════════════════════

async def get_sportapi_fixtures(date: str) -> list[dict]:
    """
    Fetch fixtures for a given date from SportAPI (141 ms, very fast).
    date: "YYYY-MM-DD"
    Falls back to Free API if SportAPI returns nothing.
    """
    cache_key = f"sportapi_fixtures_{date}"
    cached = _cache_get(cache_key, _FIXTURES_TTL)
    if cached is not None:
        return cached

    data = await _get(SPORT_API_HOST, f"/Football/FixturesByDate/{date}")
    matches = _unwrap_list(data)
    if matches:
        _cache_set(cache_key, matches)
        return matches

    # Fallback: try Free API Live Football Data
    return await get_free_api_fixtures(date)


async def get_sportapi_live_scores() -> list[dict]:
    """Fetch live scores from SportAPI. Falls back to Free API on failure."""
    cache_key = "sportapi_live"
    cached = _cache_get(cache_key, _LIVE_TTL)
    if cached is not None:
        return cached

    data = await _get(SPORT_API_HOST, "/Football/LiveScores")
    matches = _unwrap_list(data)
    if matches:
        _cache_set(cache_key, matches)
        return matches

    return await get_free_api_live_scores()
