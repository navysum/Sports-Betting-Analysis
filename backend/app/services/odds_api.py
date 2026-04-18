"""
The Odds API client — https://the-odds-api.com

Free tier: 500 requests/month. Used to fetch live bookmaker odds for upcoming
matches so value bets can be detected automatically.

Set ODDS_API_KEY in your .env file. If not set, the system falls back to
no odds (value bet detection disabled).

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
import httpx
import json
import os
import time
from typing import Optional
from app.config import settings

BASE_URL = "https://api.the-odds-api.com/v4"

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

# Cache config — 6 hours to stretch 500 req/month budget
ODDS_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "odds_cache.json"
)
CACHE_TTL = 6 * 3600  # 6 hours


def _load_odds_cache() -> dict:
    try:
        with open(ODDS_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_odds_cache(cache: dict):
    os.makedirs(os.path.dirname(ODDS_CACHE_PATH), exist_ok=True)
    with open(ODDS_CACHE_PATH, "w") as f:
        json.dump(cache, f)


async def get_pinnacle_odds(competition_code: str) -> list[dict]:
    """
    Fetch Pinnacle-only odds — the sharp market reference for CLV calculation.

    Pinnacle is the gold-standard sharp book: their closing line is the most
    efficient price in the market. Comparing model probability to Pinnacle's
    implied probability is the real edge measure.

    Returns same structure as get_odds_for_competition() but Pinnacle only.
    Cached separately (same 6h TTL but keyed differently).
    """
    global _requests_remaining

    if not settings.odds_api_key:
        return []

    sport = SPORT_MAP.get(competition_code)
    if not sport:
        return []

    cache = _load_odds_cache()
    cache_key = f"{competition_code}_pinnacle"
    cached = cache.get(cache_key)
    if cached and time.time() - cached.get("ts", 0) < CACHE_TTL:
        return cached["data"]

    # FIX #5: fetch both h2h and totals markets so over2.5 odds are available
    # for CLV tracking. Previously only h2h was fetched, making over25/btts/over35
    # CLV entries log None for implied probability — rendering them useless.
    params = {
        "apiKey":     settings.odds_api_key,
        "regions":    "eu",
        "markets":    "h2h,totals",
        "oddsFormat": "decimal",
        "bookmakers": "pinnacle",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{BASE_URL}/sports/{sport}/odds", params=params)
            resp.raise_for_status()
            _requests_remaining = int(resp.headers.get("x-requests-remaining", -1))
            events = resp.json()
    except Exception as e:
        print(f"[odds_api] Pinnacle {competition_code} fetch failed: {e}")
        return []

    results = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        odds = _extract_best_odds(event.get("bookmakers", []), home, away)
        if odds:
            results.append({
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "odds": odds,
            })

    cache[cache_key] = {"data": results, "ts": time.time()}
    _save_odds_cache(cache)
    return results


async def find_pinnacle_odds(
    home_team: str,
    away_team: str,
    competition_code: str,
) -> Optional[dict]:
    """Return Pinnacle odds for a specific match, or None."""
    events = await get_pinnacle_odds(competition_code)
    if not events:
        return None
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    for event in events:
        if _fuzzy_match(home_lower, event["home_team"].lower()) and \
           _fuzzy_match(away_lower, event["away_team"].lower()):
            return event["odds"]
    return None


async def get_odds_for_competition(competition_code: str, region: str = "uk") -> list[dict]:
    """
    Fetch odds for all upcoming matches in a competition.

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
    Cached for 6 hours to preserve the 500 req/month free tier budget.
    """
    global _requests_remaining

    if not settings.odds_api_key:
        return []

    sport = SPORT_MAP.get(competition_code)
    if not sport:
        return []

    # Check cache first
    cache = _load_odds_cache()
    cache_key = f"{competition_code}_{region}"
    cached = cache.get(cache_key)
    if cached and time.time() - cached.get("ts", 0) < CACHE_TTL:
        return cached["data"]

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
        return []

    results = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        odds = _extract_best_odds(event.get("bookmakers", []), home, away)
        if odds:
            results.append({
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "odds": odds,
            })

    # Cache the results
    cache[cache_key] = {"data": results, "ts": time.time()}
    _save_odds_cache(cache)

    return results


def _extract_best_odds(
    bookmakers: list[dict],
    home_team: str = "",
    away_team: str = "",
) -> Optional[dict]:
    """
    Take the best (highest) odds across all bookmakers for each outcome.

    Extracts:
      h2h market   → home / draw / away
      totals market → over25 / under25 (over/under 2.5 goals)

    FIX #5: previously only h2h was parsed, so over25 was always None in the
    returned dict. CLV tracking for goals markets was therefore broken — every
    over25/over35 CLV entry logged None as the implied probability, making the
    stats meaningless. Now Pinnacle's totals market (fetched in the same API
    request) is also parsed and over25 is included in the result.
    """
    best = {"home": None, "draw": None, "away": None, "over25": None, "under25": None}

    for bm in bookmakers:
        for market in bm.get("markets", []):
            mkey = market.get("key", "")
            outcomes = market.get("outcomes", [])
            if not outcomes:
                continue

            if mkey == "h2h":
                for outcome in outcomes:
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if price is None:
                        continue
                    name_lower = name.lower()
                    if name_lower == "draw":
                        key = "draw"
                    elif home_team and _fuzzy_match(name_lower, home_team.lower()):
                        key = "home"
                    elif away_team and _fuzzy_match(name_lower, away_team.lower()):
                        key = "away"
                    else:
                        continue
                    if best[key] is None or price > best[key]:
                        best[key] = price

            elif mkey == "totals":
                # The Odds API returns totals as {"name": "Over", "point": 2.5, "price": 1.85}
                for outcome in outcomes:
                    name  = outcome.get("name", "").lower()
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if price is None or point is None:
                        continue
                    # Only extract the 2.5 line
                    if abs(float(point) - 2.5) < 0.01:
                        if name == "over":
                            if best["over25"] is None or price > best["over25"]:
                                best["over25"] = price
                        elif name == "under":
                            if best["under25"] is None or price > best["under25"]:
                                best["under25"] = price

    # Return only if we have at minimum the 1X2 prices
    if all(best[k] is not None for k in ("home", "draw", "away")):
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
        # Fuzzy: check if any word from the search name appears in the event name
        if _fuzzy_match(home_lower, eh) and _fuzzy_match(away_lower, ea):
            return event["odds"]

    return None


def _fuzzy_match(search: str, target: str) -> bool:
    """True if any word in search (≥4 chars) appears in target."""
    for word in search.split():
        if len(word) >= 4 and word in target:
            return True
    # Also try whole name
    return search in target or target in search


def requests_remaining() -> Optional[int]:
    return _requests_remaining
