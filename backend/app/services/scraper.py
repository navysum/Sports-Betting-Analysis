"""
Free data scrapers for xG and historical odds.

Sources:
  - Understat.com  — xG data embedded as JSON in page scripts
  - Football-Data.co.uk — historical CSV files with odds history
"""
import re
import json
import asyncio
import io
import os
from typing import Optional
import httpx
import pandas as pd

from app.config import settings

UNDERSTAT_BASE = "https://understat.com"
FDCO_BASE = "https://www.football-data.co.uk/mmz4281"

# Understat league slugs
UNDERSTAT_LEAGUES = {
    "PL":  "EPL",
    "PD":  "La_liga",
    "BL1": "Bundesliga",
    "SA":  "Serie_A",
    "FL1": "Ligue_1",
    "ELC": "EPL",   # Championship not on Understat — fall back to EPL data
}

# Football-Data.co.uk league codes (division files)
FDCO_LEAGUES = {
    "PL":  "E0",    # Premier League
    "ELC": "E1",    # Championship
    "PD":  "SP1",   # La Liga
    "BL1": "D1",    # Bundesliga
    "SA":  "I1",    # Serie A
    "FL1": "F1",    # Ligue 1
    "DED": "N1",    # Eredivisie
    "PPL": "P1",    # Primeira Liga
}


async def fetch_understat_xg(league_code: str, season: str = "2024") -> list[dict]:
    """
    Scrape team-level xG stats from Understat for a given league/season.
    Returns a list of match dicts with keys: home, away, xG_home, xG_away,
    goals_home, goals_away, date.
    season: "2024" = 2024/25 campaign.
    """
    understat_slug = UNDERSTAT_LEAGUES.get(league_code)
    if not understat_slug:
        return []

    url = f"{UNDERSTAT_BASE}/league/{understat_slug}/{season}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SoccerBetBot/1.0; research purposes)",
        "Accept-Language": "en-GB,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        # Understat embeds data as: var datesData = JSON.parse('...')
        match = re.search(r"var datesData\s*=\s*JSON\.parse\('(.+?)'\)", html)
        if not match:
            return []

        raw = match.group(1)
        # Understat escapes the JSON — unescape unicode + single-quote wrapping
        raw = raw.encode("utf-8").decode("unicode_escape")
        data = json.loads(raw)

        results = []
        for entry in data:
            try:
                results.append({
                    "home": entry["h"]["title"],
                    "away": entry["a"]["title"],
                    "xg_home": float(entry["xG"]["h"]),
                    "xg_away": float(entry["xG"]["a"]),
                    "goals_home": int(entry["goals"]["h"]),
                    "goals_away": int(entry["goals"]["a"]),
                    "date": entry["datetime"][:10],
                    "league": league_code,
                    "season": season,
                })
            except (KeyError, ValueError, TypeError):
                continue

        return results

    except Exception as e:
        print(f"[scraper] Understat {league_code}/{season} failed: {e}")
        return []


async def fetch_understat_team_xg(team_name: str, league_code: str, season: str = "2024") -> dict:
    """
    Returns xG summary for a specific team from Understat data.
    {xg_for_avg, xg_against_avg, last5_xg_for, last5_xg_against, matches_count}
    """
    matches = await fetch_understat_xg(league_code, season)
    team_matches = []

    for m in matches:
        home_match = m["home"].lower()
        away_match = m["away"].lower()
        search = team_name.lower()

        if search in home_match or home_match in search:
            team_matches.append({"xg_for": m["xg_home"], "xg_against": m["xg_away"], "date": m["date"]})
        elif search in away_match or away_match in search:
            team_matches.append({"xg_for": m["xg_away"], "xg_against": m["xg_home"], "date": m["date"]})

    if not team_matches:
        return {}

    team_matches.sort(key=lambda x: x["date"])
    last5 = team_matches[-5:]

    return {
        "xg_for_avg": round(sum(x["xg_for"] for x in team_matches) / len(team_matches), 3),
        "xg_against_avg": round(sum(x["xg_against"] for x in team_matches) / len(team_matches), 3),
        "last5_xg_for": round(sum(x["xg_for"] for x in last5) / len(last5), 3),
        "last5_xg_against": round(sum(x["xg_against"] for x in last5) / len(last5), 3),
        "matches_count": len(team_matches),
    }


async def download_fdco_csv(league_code: str, season: str = "2425") -> Optional[pd.DataFrame]:
    """
    Download historical match CSV from Football-Data.co.uk.
    season: "2425" = 2024/25, "2324" = 2023/24, etc.
    Includes: match result, goals, and bookmaker odds columns.
    """
    division = FDCO_LEAGUES.get(league_code)
    if not division:
        return None

    url = f"{FDCO_BASE}/{season}/{division}.csv"
    out_path = os.path.join(settings.csv_dir, f"{league_code}_{season}.csv")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; SoccerBetBot/1.0)"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            content = resp.content

        # Save locally
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(content)

        df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
        df = df.dropna(how="all")
        print(f"[scraper] Downloaded {league_code} {season}: {len(df)} rows -> {out_path}")
        return df

    except Exception as e:
        print(f"[scraper] FDCO {league_code}/{season} failed: {e}")
        return None


def load_fdco_csv(league_code: str, season: str = "2425") -> Optional[pd.DataFrame]:
    """Load a previously downloaded FDCO CSV from disk."""
    path = os.path.join(settings.csv_dir, f"{league_code}_{season}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="latin-1")
        return df.dropna(how="all")
    except Exception as e:
        print(f"[scraper] load_fdco_csv error: {e}")
        return None


def extract_fdco_features(df: pd.DataFrame, home_team: str, away_team: str) -> dict:
    """
    Extract relevant features for a matchup from an FDCO CSV.
    Returns dict with: avg_home_odds, avg_draw_odds, avg_away_odds, implied_home_prob,
    implied_draw_prob, implied_away_prob.
    """
    if df is None or df.empty:
        return {}

    # Odds columns vary by bookmaker; use B365 (Bet365) as primary
    odds_cols = {"home": "B365H", "draw": "B365D", "away": "B365A"}
    fallback = {"home": "BWH", "draw": "BWD", "away": "BWA"}

    def col(key):
        return odds_cols[key] if odds_cols[key] in df.columns else fallback.get(key, "")

    result = {}
    for k in ("home", "draw", "away"):
        c = col(k)
        if c and c in df.columns:
            series = df[c].dropna()
            if not series.empty:
                avg = float(series.mean())
                result[f"avg_{k}_odds"] = round(avg, 3)
                result[f"implied_{k}_prob"] = round(1 / avg, 4) if avg > 0 else None

    # Historical performance in this matchup
    mask = (
        df["HomeTeam"].str.lower().str.contains(home_team.lower(), na=False) &
        df["AwayTeam"].str.lower().str.contains(away_team.lower(), na=False)
    )
    h2h = df[mask]
    if not h2h.empty and "FTR" in h2h.columns:
        result["fdco_h2h_home_rate"] = round((h2h["FTR"] == "H").sum() / len(h2h), 3)
        result["fdco_h2h_draw_rate"] = round((h2h["FTR"] == "D").sum() / len(h2h), 3)
        result["fdco_h2h_away_rate"] = round((h2h["FTR"] == "A").sum() / len(h2h), 3)
        result["fdco_h2h_matches"] = len(h2h)

    return result


async def bulk_download_historical(seasons: list[str] = None) -> None:
    """Download all priority leagues for the last 3 seasons."""
    if seasons is None:
        seasons = ["2425", "2324", "2223"]

    for season in seasons:
        for league_code in FDCO_LEAGUES:
            await download_fdco_csv(league_code, season)
            await asyncio.sleep(2)  # be polite
