"""
ELO rating system for football teams.

Builds from FDCO historical data, updating match-by-match in chronological order.
Ratings are persisted to data/elo_ratings.json so they survive bot restarts.

Why ELO?
  - Captures team strength across matches in a single number
  - Updates dynamically after each result
  - Goal-difference multiplier: bigger wins = bigger rating swing
  - Home advantage baked in (+100 points effective rating for home team)
  - Enables cross-league strength comparison for European fixtures

Fuzzy name matching:
  At inference time, team names from football-data.org API often differ from
  FDCO names (e.g. "Manchester Utd" vs "Man United"). EloSystem.get_rating()
  tries exact match first, then checks data/team_aliases.json, then falls
  back to difflib fuzzy matching (cutoff 0.75). If no match is found, returns
  DEFAULT_ELO (neutral) rather than silently returning 0-diff.

Usage:
    from ml.elo import EloSystem, load_elo_ratings, save_elo_ratings
    elo = load_elo_ratings()
    diff = elo.get_diff("Arsenal", "Chelsea")   # home_elo - away_elo (clamped)
"""
import difflib
import json
import math
import os
from typing import Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ELO_PATH     = os.path.join(_DATA_DIR, "elo_ratings.json")
ALIASES_PATH = os.path.join(_DATA_DIR, "team_aliases.json")

DEFAULT_ELO    = 1500.0
K_FACTOR       = 32.0          # Sensitivity to each result
HOME_ADVANTAGE = 100.0         # Points added to home team's effective rating
MAX_DIFF       = 600.0         # Clamp on elo_diff feature for normalisation
FUZZY_CUTOFF   = 0.75          # difflib similarity cutoff for name matching


def _load_aliases() -> dict[str, str]:
    """Load manual team name aliases from data/team_aliases.json.

    Keys are the names used by football-data.org API.
    Values are the names used in FDCO CSV data (which ELO is trained on).
    Returns empty dict if file is absent.
    """
    if not os.path.exists(ALIASES_PATH):
        return {}
    try:
        with open(ALIASES_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _normalise(name: str) -> str:
    """Strip common suffixes and lowercase for comparison."""
    return (
        name.lower()
        .replace(" fc", "").replace(" afc", "").replace(" cf", "")
        .replace(" utd", " united").strip()
    )


class EloSystem:
    def __init__(self):
        self._ratings: dict[str, float] = {}
        self._aliases: dict[str, str] = _load_aliases()
        # Reverse alias map: FDCO name → canonical API name (unused directly,
        # but we keep it for introspection)
        self._norm_index: dict[str, str] | None = None  # built lazily

    def _build_norm_index(self) -> dict[str, str]:
        """Build a normalised-name → original-name lookup for fuzzy matching."""
        return {_normalise(k): k for k in self._ratings}

    def resolve(self, name: str) -> str:
        """
        Map a team name to the closest name in self._ratings.

        Resolution order:
          1. Exact match
          2. Manual alias (team_aliases.json)
          3. Normalised-string exact match (strips FC/AFC/UTD)
          4. difflib fuzzy match (cutoff FUZZY_CUTOFF)
          5. Original name (caller gets DEFAULT_ELO)
        """
        if name in self._ratings:
            return name

        # Manual alias
        aliased = self._aliases.get(name)
        if aliased and aliased in self._ratings:
            return aliased

        # Normalised exact match
        norm = _normalise(name)
        if self._norm_index is None:
            self._norm_index = self._build_norm_index()
        if norm in self._norm_index:
            return self._norm_index[norm]

        # difflib fuzzy
        candidates = list(self._ratings.keys())
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=FUZZY_CUTOFF)
        if matches:
            return matches[0]
        # Also try normalised fuzzy
        norm_matches = difflib.get_close_matches(norm, list(self._norm_index.keys()), n=1, cutoff=FUZZY_CUTOFF)
        if norm_matches:
            return self._norm_index[norm_matches[0]]

        return name  # no match found → caller uses DEFAULT_ELO

    def get_rating(self, team: str) -> float:
        resolved = self.resolve(team)
        return self._ratings.get(resolved, DEFAULT_ELO)

    def get_diff(self, home_team: str, away_team: str) -> float:
        """Return (home_elo - away_elo), clamped to [-MAX_DIFF, MAX_DIFF]."""
        diff = self.get_rating(home_team) - self.get_rating(away_team)
        return max(-MAX_DIFF, min(MAX_DIFF, diff))

    def _expected(self, rating_a: float, rating_b: float) -> float:
        """Standard ELO expected score for A playing B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def process_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        """Update ELO ratings after a match result."""
        # Home team has effective rating boost for home advantage
        r_home_eff = self.get_rating(home_team) + HOME_ADVANTAGE
        r_away_eff = self.get_rating(away_team)

        e_home = self._expected(r_home_eff, r_away_eff)
        e_away = 1.0 - e_home

        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals == away_goals:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        # Goal-difference multiplier: bigger margins → larger rating swing
        gd = abs(home_goals - away_goals)
        gd_mult = math.log(gd + 1) + 1.0 if gd > 0 else 1.0

        k = K_FACTOR * gd_mult
        # Write back using original names (process_match is always called with
        # FDCO names, so no resolution needed here)
        prev_home = self._ratings.get(home_team, DEFAULT_ELO)
        prev_away = self._ratings.get(away_team, DEFAULT_ELO)
        self._ratings[home_team] = prev_home + k * (s_home - e_home)
        self._ratings[away_team] = prev_away + k * (s_away - e_away)
        # Invalidate norm index after new team added
        self._norm_index = None

    def to_dict(self) -> dict:
        return dict(self._ratings)

    @classmethod
    def from_dict(cls, data: dict) -> "EloSystem":
        elo = cls()
        elo._ratings = {k: float(v) for k, v in data.items()}
        return elo


def save_elo_ratings(elo: EloSystem) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(ELO_PATH, "w") as f:
        json.dump(elo.to_dict(), f, indent=2)
    print(f"  [elo] Saved {len(elo._ratings)} team ratings → {ELO_PATH}")


def load_elo_ratings() -> EloSystem:
    if not os.path.exists(ELO_PATH):
        return EloSystem()
    try:
        with open(ELO_PATH) as f:
            data = json.load(f)
        elo = EloSystem.from_dict(data)
        return elo
    except Exception:
        return EloSystem()
