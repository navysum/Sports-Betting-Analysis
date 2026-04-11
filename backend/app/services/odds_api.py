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
    """
    global _requests_remaining

    if not settings.odds_api_key:
        return []

    sport = SPORT_MAP.get(competition_code)
    if not sport:
        return []

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
        odds = _extract_best_odds(event.get("bookmakers", []))
        if odds:
            results.append({
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "odds": odds,
            })

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
            if len(outcomes) < 2:
                continue

            for outcome in outcomes:
                name = outcome.get("name", "")
                price = outcome.get("price")
                if price is None:
                    continue

                # Map outcome name → home/draw/away based on position
                # The Odds API labels outcomes by team name, not home/away
                # We store them by position in the outcomes list
                if len(outcomes) == 3:
                    # h2h with draw
                    if outcome == outcomes[0]:
                        key = "home"
                    elif outcome == outcomes[1]:
                        key = "draw"
                    else:
                        key = "away"
                else:
                    continue  # No draw — skip (shouldn't happen for football)

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
