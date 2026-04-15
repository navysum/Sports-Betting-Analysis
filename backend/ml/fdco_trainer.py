"""
Build training data from Football-Data.co.uk CSV files.

These CSVs cover multiple seasons and include Bet365 odds, giving us 3,000+
samples without additional API calls. They are downloaded by scraper.py into
data/csv/ and cached locally.

Usage:
    from ml.fdco_trainer import build_fdco_training_data
    X, y_result, y_goals, y_btts, odds_rows = build_fdco_training_data()

The returned odds_rows are used by the backtesting module.
"""
import hashlib
import os
import io
import asyncio
import numpy as np
import pandas as pd
from typing import Optional

from app.config import settings
from ml.features import build_feature_vector, N_FEATURES
from ml.elo import EloSystem, save_elo_ratings

# Seasons to load — chronological order for correct form/standings accumulation.
# Extended back to 2010/11 (~15 seasons, ~3× more data than before).
# Download silently skips missing files, so adding extra seasons is safe.
SEASONS = [
    "1011", "1112", "1213", "1314", "1415", "1516", "1617",
    "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425",
]

# football-data.co.uk season codes  (URL path segment)
FDCO_SEASON_CODES = {s: s for s in SEASONS}

# League CSV division codes
FDCO_LEAGUES = {
    "PL":  "E0",
    "ELC": "E1",
    "PD":  "SP1",
    "BL1": "D1",
    "SA":  "I1",
    "FL1": "F1",
    "DED": "N1",
    "PPL": "P1",
}

# Number of teams per league — used to normalise standing position
LEAGUE_SIZES = {
    "PL": 20, "ELC": 24, "PD": 20, "BL1": 18,
    "SA": 20, "FL1": 20, "DED": 18, "PPL": 18,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _team_id(name: str) -> int:
    """Stable, deterministic integer ID from a team name."""
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def _parse_date(raw: str) -> Optional[str]:
    """Convert DD/MM/YYYY or DD/MM/YY → YYYY-MM-DD. Returns None on failure."""
    raw = raw.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            from datetime import datetime
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _rows_to_match_dicts(df: pd.DataFrame) -> list[dict]:
    """
    Convert a FDCO DataFrame to a list of match dicts compatible with the
    feature-engineering functions (same shape as football-data.org API responses).
    """
    records = []
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if not required.issubset(df.columns):
        return records

    for _, row in df.iterrows():
        try:
            home = str(row["HomeTeam"]).strip()
            away = str(row["AwayTeam"]).strip()
            hg   = int(row["FTHG"])
            ag   = int(row["FTAG"])
            raw_date = str(row.get("Date", "")).strip()
            date_iso = _parse_date(raw_date) or "1970-01-01"

            records.append({
                "status": "FINISHED",
                "utcDate": date_iso + "T15:00:00Z",
                "homeTeam": {"id": _team_id(home), "name": home},
                "awayTeam": {"id": _team_id(away), "name": away},
                "score": {"fullTime": {"home": hg, "away": ag}},
                # Odds (for backtesting only — not used in feature vector)
                "_b365h": _safe_float(row, "B365H"),
                "_b365d": _safe_float(row, "B365D"),
                "_b365a": _safe_float(row, "B365A"),
                "_home": home,
                "_away": away,
                "_date": date_iso,
            })
        except (ValueError, TypeError):
            continue
    return records


def _safe_float(row, col: str) -> Optional[float]:
    try:
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


# ─── Running standings ────────────────────────────────────────────────────────

class _RunningTable:
    """Maintains a live league table that updates match by match."""

    def __init__(self):
        self._table: dict[str, dict] = {}  # team_name → stats

    def _ensure(self, team: str):
        if team not in self._table:
            self._table[team] = {"points": 0, "played": 0, "gd": 0, "position": 10}

    def update(self, home: str, away: str, hg: int, ag: int):
        self._ensure(home); self._ensure(away)
        self._table[home]["played"] += 1
        self._table[away]["played"] += 1
        self._table[home]["gd"] += hg - ag
        self._table[away]["gd"] += ag - hg
        if hg > ag:
            self._table[home]["points"] += 3
        elif hg == ag:
            self._table[home]["points"] += 1
            self._table[away]["points"] += 1
        else:
            self._table[away]["points"] += 3
        # Recompute positions
        sorted_teams = sorted(
            self._table.keys(),
            key=lambda t: (self._table[t]["points"], self._table[t]["gd"]),
            reverse=True,
        )
        for rank, t in enumerate(sorted_teams, 1):
            self._table[t]["position"] = rank

    def standing(self, team: str) -> Optional[dict]:
        if team not in self._table:
            return None
        s = self._table[team]
        return {
            "position": s["position"],
            "points": s["points"],
            "playedGames": max(s["played"], 1),
        }


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_fdco_training_data(
    min_history: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Load all cached FDCO CSVs and build feature vectors.

    Returns:
        X         — float32 feature matrix  (N, N_FEATURES)
        y_result  — int array  0=HOME, 1=DRAW, 2=AWAY
        y_goals   — int array  0=Under2.5, 1=Over2.5
        y_btts    — int array  0=No, 1=Yes
        y_over35  — int array  0=Under3.5, 1=Over3.5
        odds_rows — list of dicts for backtesting  {date, home, away, b365h, b365d, b365a, ...}
    """
    csv_dir = settings.csv_dir
    X_all, y_result_all, y_goals_all, y_btts_all, y_over35_all = [], [], [], [], []
    odds_rows = []
    total_files = 0

    # Per-league ELO systems that persist across seasons (team strength is continuous)
    league_elo: dict[str, EloSystem] = {}

    for league_code, division in FDCO_LEAGUES.items():
        total_teams = LEAGUE_SIZES.get(league_code, 20)
        # Fresh ELO for each league — cross-league mixing not meaningful for domestic
        elo = EloSystem()
        league_elo[league_code] = elo

        for season in SEASONS:
            path = os.path.join(csv_dir, f"{league_code}_{season}.csv")
            if not os.path.exists(path):
                continue

            try:
                df = pd.read_csv(path, encoding="latin-1").dropna(how="all")
            except Exception as e:
                print(f"  [fdco] Failed to load {path}: {e}")
                continue

            match_dicts = _rows_to_match_dicts(df)
            if not match_dicts:
                continue

            total_files += 1
            # Sort by date for sequential form/standings computation
            match_dicts.sort(key=lambda m: m.get("utcDate", ""))

            table = _RunningTable()
            team_history: dict[int, list[dict]] = {}  # team_id → match list

            for m in match_dicts:
                home_name = m["homeTeam"]["name"]
                away_name = m["awayTeam"]["name"]
                home_id   = m["homeTeam"]["id"]
                away_id   = m["awayTeam"]["id"]
                hg = m["score"]["fullTime"]["home"]
                ag = m["score"]["fullTime"]["away"]
                date_str = m.get("utcDate", "")[:10]

                home_hist = team_history.get(home_id, [])
                away_hist = team_history.get(away_id, [])

                # Need minimum history to produce reliable features
                if len(home_hist) < min_history or len(away_hist) < min_history:
                    # Still update history and table for next matches
                    table.update(home_name, away_name, hg, ag)
                    team_history.setdefault(home_id, []).append(m)
                    team_history.setdefault(away_id, []).append(m)
                    continue

                # Labels
                if hg > ag:    result_label = 0
                elif hg == ag: result_label = 1
                else:          result_label = 2

                goals_label  = 1 if (hg + ag) > 2 else 0
                btts_label   = 1 if (hg > 0 and ag > 0) else 0
                over35_label = 1 if (hg + ag) > 3 else 0

                # Pre-match ELO difference (computed BEFORE updating ELO)
                elo_diff = elo.get_diff(home_name, away_name)

                vec = build_feature_vector(
                    home_id=home_id,
                    away_id=away_id,
                    home_matches=home_hist[-25:],
                    away_matches=away_hist[-25:],
                    h2h_matches=[
                        mm for mm in home_hist
                        if mm["awayTeam"]["id"] == away_id or mm["homeTeam"]["id"] == away_id
                    ][-10:],
                    home_standing=table.standing(home_name),
                    away_standing=table.standing(away_name),
                    match_date=date_str,
                    elo_diff=elo_diff,
                    total_teams=total_teams,
                    # Odds intentionally excluded from features — kept in odds_rows for backtesting only
                )

                X_all.append(vec)
                y_result_all.append(result_label)
                y_goals_all.append(goals_label)
                y_btts_all.append(btts_label)
                y_over35_all.append(over35_label)

                # Record odds for backtesting + DC model building
                odds_rows.append({
                    "date": date_str,
                    "league": league_code,
                    "home": home_name,
                    "away": away_name,
                    "home_goals": hg,
                    "away_goals": ag,
                    "result_label": result_label,
                    "goals_label": goals_label,
                    "btts_label": btts_label,
                    "b365h": m.get("_b365h"),
                    "b365d": m.get("_b365d"),
                    "b365a": m.get("_b365a"),
                    "feature_idx": len(X_all) - 1,  # index into X
                })

                # Update running state AFTER feature vector is built (no future leakage)
                table.update(home_name, away_name, hg, ag)
                team_history.setdefault(home_id, []).append(m)
                team_history.setdefault(away_id, []).append(m)
                elo.process_match(home_name, away_name, hg, ag)

    if not X_all:
        print(f"  [fdco] No CSV data found in {csv_dir}. Run scraper first.")
        return (
            np.empty((0, N_FEATURES), dtype=np.float32),
            np.array([]), np.array([]), np.array([]), np.array([]),
            [],
        )

    # Persist merged ELO ratings (most recent per-league ratings saved globally)
    # Teams in different leagues keep separate histories, merged here for inference
    merged_elo = EloSystem()
    for elo in league_elo.values():
        for team, rating in elo._ratings.items():
            # If same team name in multiple leagues (rare), keep higher rating
            if team not in merged_elo._ratings or rating > merged_elo._ratings[team]:
                merged_elo._ratings[team] = rating
    save_elo_ratings(merged_elo)

    print(f"  [fdco] {total_files} CSV files -> {len(X_all)} training samples")
    return (
        np.array(X_all, dtype=np.float32),
        np.array(y_result_all),
        np.array(y_goals_all),
        np.array(y_btts_all),
        np.array(y_over35_all),
        odds_rows,
    )


async def download_all_csvs(seasons: list[str] = None) -> None:
    """Download FDCO CSVs for all leagues and seasons."""
    from app.services.scraper import download_fdco_csv
    if seasons is None:
        seasons = list(reversed(SEASONS))  # newest first
    current_season = seasons[0]  # newest = still in progress, always refresh
    print(f"  [fdco] Downloading CSVs for {len(FDCO_LEAGUES)} leagues × {len(seasons)} seasons…")
    for season in seasons:
        for league_code in FDCO_LEAGUES:
            csv_path = os.path.join(settings.csv_dir, f"{league_code}_{season}.csv")
            if season != current_season and os.path.exists(csv_path):
                continue  # historical season already cached
            await download_fdco_csv(league_code, season)
            await asyncio.sleep(1)
    print("  [fdco] Download complete.")
