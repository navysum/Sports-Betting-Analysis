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

Usage:
    from ml.elo import EloSystem, load_elo_ratings, save_elo_ratings
    elo = load_elo_ratings()
    diff = elo.get_diff("Arsenal", "Chelsea")   # home_elo - away_elo (clamped)
"""
import json
import math
import os
from typing import Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ELO_PATH = os.path.join(_DATA_DIR, "elo_ratings.json")

DEFAULT_ELO = 1500.0
K_FACTOR = 32.0          # Sensitivity to each result
HOME_ADVANTAGE = 100.0   # Points added to home team's effective rating
MAX_DIFF = 600.0         # Clamp on elo_diff feature for normalisation


class EloSystem:
    def __init__(self):
        self._ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self._ratings.get(team, DEFAULT_ELO)

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
        self._ratings[home_team] = self.get_rating(home_team) + k * (s_home - e_home)
        self._ratings[away_team] = self.get_rating(away_team) + k * (s_away - e_away)

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
    print(f"  [elo] Saved {len(elo._ratings)} team ratings -> {ELO_PATH}")


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
