"""
FBref scraper — fetches per-team xG, xGA, and xPts data to supplement
the main prediction model with a true quality signal beyond actual results.

FBref rate-limits aggressively so all results are cached for 24 hours.
Falls back gracefully (returns {}) on any error.

Data fetched:
  - xG (expected goals for)
  - xGA (expected goals against)
  - xGD (xG - xGA, the "true quality" differential)
  - xPts (expected points based on xG each match)
  - npxGD (non-penalty xGD — more stable long-term quality signal)
"""
import asyncio
import json
import os
import time
import httpx
from typing import Optional
from bs4 import BeautifulSoup

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "fbref_cache.json",
)
CACHE_TTL = 24 * 3600  # 24 hours

# Competition code → FBref competition ID + slug
FBREF_LEAGUES = {
    "PL":  (9,  "Premier-League"),
    "PD":  (12, "La-Liga"),
    "BL1": (20, "Bundesliga"),
    "SA":  (11, "Serie-A"),
    "FL1": (13, "Ligue-1"),
    "DED": (23, "Eredivisie"),
    "ELC": (10, "Championship"),
    "PPL": (32, "Primeira-Liga"),
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://fbref.com/",
}

# Module-level in-memory layer on top of disk cache (avoids repeated file reads)
_mem: dict = {}


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


def _cache_get(key: str) -> Optional[dict]:
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


def _cache_set(key: str, data: dict):
    entry = {"data": data, "ts": time.time()}
    _mem[key] = entry
    disk = _load_cache()
    disk[key] = entry
    _save_cache(disk)


def _parse_float(val: str) -> Optional[float]:
    try:
        return float(val.strip().replace(",", ""))
    except Exception:
        return None


def _parse_squad_table(soup: BeautifulSoup, table_id: str) -> dict[str, dict]:
    """
    Parse an FBref squad stats table into a dict keyed by squad name.
    Returns {squad_name: {col: value, ...}}
    """
    table = soup.find("table", {"id": table_id})
    if not table:
        return {}

    headers = []
    thead = table.find("thead")
    if thead:
        # FBref uses two header rows — use the last one for column names
        rows = thead.find_all("tr")
        for th in rows[-1].find_all(["th", "td"]):
            stat = th.get("data-stat", th.get_text(strip=True))
            headers.append(stat)

    results = {}
    tbody = table.find("tbody")
    if not tbody:
        return {}

    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class", []):
            continue
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        row = {}
        for i, cell in enumerate(cells):
            if i < len(headers):
                row[headers[i]] = cell.get_text(strip=True)

        squad = row.get("team") or row.get("squad") or row.get("Squad")
        if squad:
            results[squad] = row

    return results


async def fetch_league_xg(competition_code: str) -> dict[str, dict]:
    """
    Fetch xG stats for all teams in a competition.

    Returns {team_name: {xg, xga, xgd, npxgd, xpts, pts, gp}, ...}
    Empty dict on error.
    """
    cached = _cache_get(competition_code)
    if cached is not None:
        return cached

    league_info = FBREF_LEAGUES.get(competition_code)
    if not league_info:
        return {}

    comp_id, slug = league_info
    url = f"https://fbref.com/en/comps/{comp_id}/{slug}-Stats"

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        print(f"[fbref] {competition_code}: fetch failed — {e}")
        return {}

    soup = BeautifulSoup(html, "lxml")

    # FBref standard squad stats table IDs vary; try both naming patterns
    table_ids = [
        f"results{comp_id}1_overall",
        "results_overall",
        f"stats_squads_standard_for",
    ]

    squad_rows: dict[str, dict] = {}
    for tid in table_ids:
        squad_rows = _parse_squad_table(soup, tid)
        if squad_rows:
            break

    if not squad_rows:
        # Fallback: find any table with an xg column
        for table in soup.find_all("table"):
            tid = table.get("id", "")
            rows = _parse_squad_table(soup, tid)
            if rows:
                sample = next(iter(rows.values()))
                if any("xg" in k.lower() for k in sample):
                    squad_rows = rows
                    break

    if not squad_rows:
        print(f"[fbref] {competition_code}: no squad table found")
        return {}

    result = {}
    for squad, row in squad_rows.items():
        # Try multiple column name variants (FBref changes them occasionally)
        xg   = _parse_float(row.get("xg") or row.get("xG") or row.get("expected_goals") or "")
        xga  = _parse_float(row.get("xga") or row.get("xGA") or row.get("expected_goals_against") or "")
        npxg = _parse_float(row.get("npxg") or row.get("npxG") or row.get("expected_goals_np") or "")
        npxga= _parse_float(row.get("npxga") or row.get("npxGA") or "")
        xpts = _parse_float(row.get("xpts") or row.get("xPts") or row.get("expected_points") or "")
        pts  = _parse_float(row.get("pts") or row.get("Pts") or row.get("points") or "")
        gp   = _parse_float(row.get("mp") or row.get("MP") or row.get("games") or "")

        if xg is None and xga is None:
            continue

        xgd    = (xg or 0.0) - (xga or 0.0)
        npxgd  = (npxg or xg or 0.0) - (npxga or xga or 0.0)
        pts_over_xpts = (pts - xpts) / max(gp, 1) if pts is not None and xpts is not None and gp else 0.0

        result[squad] = {
            "xg":             xg or 0.0,
            "xga":            xga or 0.0,
            "xgd":            xgd,
            "npxgd":          npxgd,
            "xpts":           xpts,
            "pts":            pts,
            "gp":             gp,
            "pts_over_xpts":  pts_over_xpts,  # positive = overperforming, negative = underperforming
        }

    if result:
        _cache_set(competition_code, result)
        print(f"[fbref] {competition_code}: {len(result)} teams cached")

    return result


def _fuzzy_find(name: str, squad_map: dict[str, dict]) -> Optional[dict]:
    """Case-insensitive partial-match lookup for team names."""
    name_l = name.lower().strip()
    # Exact first
    for k, v in squad_map.items():
        if k.lower() == name_l:
            return v
    # Word overlap — longest overlap wins
    best_key, best_score = None, 0
    for k in squad_map:
        kl = k.lower()
        overlap = sum(1 for w in name_l.split() if len(w) >= 3 and w in kl)
        if overlap > best_score:
            best_score, best_key = overlap, k
    if best_score > 0:
        return squad_map[best_key]
    return None


async def get_team_xg_stats(
    team_name: str,
    competition_code: str,
) -> dict:
    """
    Returns xG quality stats for a single team.

    {
      xg, xga, xgd, npxgd,
      pts_over_xpts,   # how many extra pts/game vs expected (regression signal)
    }
    Returns {} if unavailable.
    """
    league_data = await fetch_league_xg(competition_code)
    if not league_data:
        return {}
    found = _fuzzy_find(team_name, league_data)
    return found or {}
