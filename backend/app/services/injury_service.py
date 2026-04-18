"""
Injury and suspension data via API-Football (RapidAPI).

Free tier: 100 requests/day. Results are cached per day to preserve quota.
Falls back gracefully (returns []) when key is absent or quota is exhausted.

Usage:
    injuries = await get_team_injuries(team_name, competition_code, match_date)
    # [{"player": "...", "type": "Injury|Suspension", "reason": "..."}, ...]
"""
import json
import os
import time
import asyncio
import httpx
from datetime import datetime
from typing import Optional

from app.config import settings

BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "injury_cache.json",
)
# Cache for the rest of the day — injuries rarely change intraday
CACHE_TTL = 12 * 3600

# Competition code → API-Football league ID + typical season year
LEAGUE_IDS = {
    "PL":  39,
    "PD":  140,
    "BL1": 78,
    "SA":  135,
    "FL1": 61,
    "ELC": 40,
    "DED": 88,
    "PPL": 94,
    "CL":  2,
}

_mem: dict = {}


def _current_season() -> int:
    """Return the current football season start year (e.g. 2025 for 2025/26)."""
    now = datetime.utcnow()
    return now.year if now.month >= 7 else now.year - 1


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(data: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f)


def _cache_get(key: str) -> Optional[list]:
    if key in _mem:
        entry = _mem[key]
        if time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]
    disk = _load_cache()
    entry = disk.get(key)
    if entry and time.time() - entry.get("ts", 0) < CACHE_TTL:
        _mem[key] = entry
        return entry["data"]
    return None


def _cache_set(key: str, data: list):
    entry = {"data": data, "ts": time.time()}
    _mem[key] = entry
    disk = _load_cache()
    disk[key] = entry
    _save_cache(disk)


async def _fetch_injuries(league_id: int, season: int, date_str: str) -> list[dict]:
    """Raw API call. Returns list of injury objects from API-Football.

    Uses settings.api_football_key if set; falls back to settings.rapidapi_key
    so the single RapidAPI key from the .env is enough for everything.
    """
    key = settings.api_football_key or settings.rapidapi_key
    if not key:
        return []

    params = {"league": league_id, "season": season, "date": date_str}
    headers = {
        "X-RapidAPI-Key":  key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{BASE_URL}/injuries", params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", [])
    except Exception as e:
        print(f"[injuries] API call failed: {e}")
        return []


async def get_all_match_injuries(
    competition_code: str,
    match_date: str,
) -> dict[str, list[dict]]:
    """
    Fetch all injuries/suspensions for a league on a given date.

    Returns {team_name: [{player, type, reason}, ...], ...}
    """
    league_id = LEAGUE_IDS.get(competition_code)
    key = settings.api_football_key or settings.rapidapi_key
    if not league_id or not key:
        return {}

    date_str = match_date[:10] if match_date else datetime.utcnow().strftime("%Y-%m-%d")
    cache_key = f"{competition_code}_{date_str}"

    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    season = _current_season()
    raw = await _fetch_injuries(league_id, season, date_str)

    if not raw:
        # Don't cache empty — might be an API error vs genuinely no injuries
        return {}

    # Group by team name
    by_team: dict[str, list] = {}
    for entry in raw:
        team_name = entry.get("team", {}).get("name", "")
        player    = entry.get("player", {}).get("name", "")
        reason    = entry.get("player", {}).get("reason", "")
        inj_type  = entry.get("player", {}).get("type", "Injury")

        if team_name and player:
            by_team.setdefault(team_name, []).append({
                "player": player,
                "type":   inj_type,
                "reason": reason,
            })

    _cache_set(cache_key, by_team)
    print(f"[injuries] {competition_code} {date_str}: {sum(len(v) for v in by_team.values())} injuries across {len(by_team)} teams")
    return by_team


def _fuzzy_find_team(name: str, team_map: dict) -> list[dict]:
    """Fuzzy match a team name against the injury map."""
    name_l = name.lower().strip()
    for k, v in team_map.items():
        kl = k.lower()
        if kl == name_l or name_l in kl or kl in name_l:
            return v
        overlap = sum(1 for w in name_l.split() if len(w) >= 4 and w in kl)
        if overlap >= 1:
            return v
    return []


async def get_team_injuries(
    team_name: str,
    competition_code: str,
    match_date: str = "",
) -> list[dict]:
    """
    Return list of injured/suspended players for a specific team.
    Each item: {"player": str, "type": "Injury"|"Suspension", "reason": str}
    """
    all_injuries = await get_all_match_injuries(competition_code, match_date)
    if not all_injuries:
        return []
    return _fuzzy_find_team(team_name, all_injuries)


def injury_adjustment(injuries: list[dict]) -> float:
    """
    Convert a list of injuries into a win-probability adjustment factor.

    Returns a negative float (penalty) — caller applies it to the team's win prob.
    Scale: -0.03 per injured player, -0.05 per suspended player.
    Capped at -0.15 to avoid overcorrecting.
    """
    adj = 0.0
    for inj in injuries:
        inj_type = inj.get("type", "Injury").lower()
        if "suspension" in inj_type:
            adj -= 0.05
        else:
            adj -= 0.03
    return max(adj, -0.15)
