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
    Uses team names to correctly assign home/draw/away — The Odds API labels
    outcomes by team name, not position, so position-based assignment was wrong.
    """
    best = {"home": None, "draw": None, "away": None}

    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 2:
                continue

            for outcome in outcomes:
                name = outcome.get("name", "")
                price = outcome.get("price")
                if price is None:
                    continue

                # Match by name: "Draw" is literal, teams matched by name
                name_lower = name.lower()
                if name_lower == "draw":
                    key = "draw"
                elif home_team and _fuzzy_match(name_lower, home_team.lower()):
                    key = "home"
                elif away_team and _fuzzy_match(name_lower, away_team.lower()):
                    key = "away"
                else:
                    continue  # Can't reliably identify this outcome

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
