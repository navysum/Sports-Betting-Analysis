from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """
    Application Settings Class.
    This uses Pydantic to manage configuration. It automatically loads values
    from environment variables or a .env file.
    """

    # --- Football API Keys ---
    football_data_api_key: str = ""
    api_football_key: str = ""
    odds_api_key: str = ""
    rapidapi_key: str = ""

    # --- Database Configuration ---
    database_url: str = "sqlite+aiosqlite:///./data/soccerbet.db"

    # --- Filesystem Paths ---
    data_dir: str = "./data"
    models_dir: str = "./ml"
    csv_dir: str = "./data/csv"

    # --- Timezone Settings ---
    timezone: str = "Europe/London"

    # ── AI Decision Layer: bet eligibility thresholds ─────────────────────────
    # Minimum edge (model_prob - fair_implied) required to flag a bet.
    min_edge_global: float = 0.05

    # Per-market minimum confidence thresholds (model probability for best outcome).
    # Tighter thresholds raise hit-rate at the cost of lower volume.
    min_confidence_1x2: float = 0.55
    min_confidence_over25: float = 0.60
    min_confidence_btts: float = 0.58
    min_confidence_over35: float = 0.62

    # Per-market minimum edge overrides
    min_edge_1x2: float = 0.06
    min_edge_over25: float = 0.05
    min_edge_btts: float = 0.05
    min_edge_over35: float = 0.06

    # ── Odds band filters ────────────────────────────────────────────────────
    # Only flag bets when bookmaker odds fall inside these ranges.
    odds_min_1x2: float = 1.30
    odds_max_1x2: float = 3.50
    odds_min_over25: float = 1.40
    odds_max_over25: float = 2.50
    odds_min_btts: float = 1.40
    odds_max_btts: float = 2.20
    odds_min_over35: float = 1.60
    odds_max_over35: float = 3.20

    # ── AI scoring weights ───────────────────────────────────────────────────
    # Must sum to 1.0
    ai_weight_edge: float = 0.35
    ai_weight_confidence: float = 0.25
    ai_weight_historical: float = 0.20
    ai_weight_clv: float = 0.10
    ai_weight_data_quality: float = 0.10

    # AI recommendation score thresholds
    ai_score_strong_bet: float = 8.5
    ai_score_bet: float = 7.0
    ai_score_small_bet: float = 6.0
    ai_score_watchlist: float = 5.0

    # ── Supported leagues for strict eligibility mode ────────────────────────
    strict_leagues: list = ["PL", "BL1", "PD", "SA", "FL1", "ELC"]

    class Config:
        env_file = ".env"

    def ensure_dirs(self):
        """Create required data directories if missing."""
        for d in [self.data_dir, self.models_dir, self.csv_dir]:
            os.makedirs(d, exist_ok=True)


# 1. Initialize the settings object
settings = Settings()

# 2. Run the directory check to make sure the app has a place to save files
settings.ensure_dirs()

