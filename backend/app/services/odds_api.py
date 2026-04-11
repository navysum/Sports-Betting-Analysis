"""
The Odds API client — https://the-odds-api.com

Free tier: 500 requests/month. Used to fetch live bookmaker odds for upcoming
matches so value bets can be detected automatically.

Set ODDS_API_KEY in your .env file. If not set, the system falls back to
no odds (value bet detection disabled).

Caching:
    Odds are cached in data/odds_cache.json keyed by {competition}:{date}.
    Cached data is considered fresh for CACHE_TTL_HOURS hours. This prevents
    re-fetching the same competition's odds multiple times per day and keeps
    usage well within the 500 req/month free tier.

Competition code → Odds API sport key mapping:
    PL   → soccer_epl
    PD   → soccer_spain_la_liga
    BL1  → soccer_germany_bundesliga
    SA   → soccer_italy_serie_a
    FL1  → soccer_france_ligue_one
    ELC  → soccer_efl_champ
    DED  → soccer_netherlands_eredivisie
"""
import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

from app.config import settings

BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_TTL_HOURS = 6

_CACHE_PATH = os.path.join(settings.data_dir, "odds_cache.json")

SPORT_MAP = {
    "PL":  "soccer_epl",
    "PD":  "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "SA":  "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_one",
    "ELC": "soccer_efl_champ",
    "DED": "soccer_netherlands_eredivisie",
    "PPL": "soccer_portugal_primeira_liga",
    "CL":  "soccer_uefa_champs_league",
}

# Requests remaining counter (updated from response headers)
_requests_remaining: Optional[int] = None


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def _cache_key(competition_code: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{competition_code}:{today}"


def _is_fresh(entry: dict) -> bool:
    """Return True if the cache entry was fetched within CACHE_TTL_HOURS."""
    try:
        fetched = datetime.fromisoformat(entry["fetched_at"])
        age = datetime.now(timezone.utc) - fetched
        return age < timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return False


# ─── API fetch ────────────────────────────────────────────────────────────────

async def get_odds_for_competition(competition_code: str, region: str = "uk") -> list[dict]:
    """
    Fetch odds for all upcoming matches in a competition.

    Returns cached data if it was fetched within CACHE_TTL_HOURS hours.
    Otherwise fetches from The Odds API and updates the cache.

    Returns a list of event dicts:
    [
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2026-04-12T14:00:00Z",
            "odds": {"home": 2.10, "draw": 3.40, "away": 3.20}
        },
        ...
    ]
    Returns [] if no API key is configured or on error.
    """
    global _requests_remaining

    if not settings.odds_api_key:
        return []

    sport = SPORT_MAP.get(competition_code)
    if not sport:
        return []

    key = _cache_key(competition_code)
    cache = _load_cache()

    # Return cached data if fresh
    if key in cache and _is_fresh(cache[key]):
        return cache[key]["events"]

    # Fetch from API
    params = {
        "apiKey": settings.odds_api_key,
        "regions": region,
        "markets": "h2h",
        "oddsFormat": "decimal",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{BASE_URL}/sports/{sport}/odds", params=params)
            resp.raise_for_status()
            _requests_remaining = int(resp.headers.get("x-requests-remaining", -1))
            events = resp.json()
    except Exception as e:
        print(f"[odds_api] {competition_code} fetch failed: {e}")
        # Return stale cache if available rather than nothing
        if key in cache:
            return cache[key]["events"]
        return []

    results = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        odds = _extract_best_odds(event.get("bookmakers", []))
        if odds:
            results.append({
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "odds": odds,
            })

    # Update cache
    cache[key] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "events": results,
    }
    # Prune stale entries (older than 2 days) to keep file small
    stale_keys = [
        k for k, v in cache.items()
        if not _is_fresh(v) and k != key
    ]
    for k in stale_keys:
        try:
            date_part = k.split(":")[1]
            entry_date = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - entry_date).days > 2:
                del cache[k]
        except Exception:
            pass
    _save_cache(cache)

    return results


def _extract_best_odds(bookmakers: list[dict]) -> Optional[dict]:
    """
    Take the best (highest) odds across all bookmakers for each outcome.
    Higher odds = better value for the bettor.
    """
    best = {"home": None, "draw": None, "away": None}

    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 3:
                continue  # Need all three outcomes for football

            for idx, outcome in enumerate(outcomes):
                price = outcome.get("price")
                if price is None:
                    continue
                if idx == 0:
                    key = "home"
                elif idx == 1:
                    key = "draw"
                else:
                    key = "away"

                if best[key] is None or price > best[key]:
                    best[key] = price

    if all(v is not None for v in best.values()):
        return best
    return None


async def find_match_odds(
    home_team: str,
    away_team: str,
    competition_code: str,
) -> Optional[dict]:
    """
    Find odds for a specific match. Does fuzzy name matching since team names
    differ between football-data.org and The Odds API.

    Returns {"home": float, "draw": float, "away": float} or None.
    """
    events = await get_odds_for_competition(competition_code)
    if not events:
        return None

    home_lower = home_team.lower()
    away_lower = away_team.lower()

    for event in events:
        eh = event["home_team"].lower()
        ea = event["away_team"].lower()
        if _fuzzy_match(home_lower, eh) and _fuzzy_match(away_lower, ea):
            return event["odds"]

    return None


def _fuzzy_match(search: str, target: str) -> bool:
    """True if any word in search (≥4 chars) appears in target."""
    for word in search.split():
        if len(word) >= 4 and word in target:
            return True
    return search in target or target in search


def requests_remaining() -> Optional[int]:
    return _requests_remaining
