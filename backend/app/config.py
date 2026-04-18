from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """
    Application Settings Class.
    This uses Pydantic to manage configuration. It automatically loads values 
    from environment variables or a .env file.
    """

    # --- Football API Keys ---
    # These are used to fetch live match data and betting odds from external services.
    football_data_api_key: str = ""
    api_football_key: str = ""       # RapidAPI free tier — 100 req/day (legacy, see rapidapi_key)
    odds_api_key: str = ""           # the-odds-api.com — 500 req/month free

    # --- RapidAPI (single key, 3 subscriptions) ---
    # Covers: Sofascore (xG), Free API Live Football Data, SportAPI
    # Set RAPIDAPI_KEY in .env or Render environment variables.
    rapidapi_key: str = ""

    # --- Database Configuration ---
    # The URL for the SQLite database. 
    # 'sqlite+aiosqlite' tells the app to use an asynchronous driver.
    database_url: str = "sqlite+aiosqlite:///./data/soccerbet.db"

    # --- Filesystem Paths ---
    # These define where the app stores its local data, ML models, and CSV files.
    # Paths are relative to the 'backend/' directory.
    data_dir: str = "./data"
    models_dir: str = "./ml"
    csv_dir: str = "./data/csv"

    # --- Timezone Settings ---
    # Used for scheduling the daily prediction updates.
    timezone: str = "Europe/London"

    class Config:
        # Tells Pydantic to look for a file named '.env' in the same directory.
        env_file = ".env"

    def ensure_dirs(self):
        """
        Utility method to ensure that all required data directories exist.
        If a directory is missing, this will create it automatically.
        """
        for d in [self.data_dir, self.models_dir, self.csv_dir]:
            os.makedirs(d, exist_ok=True)


# 1. Initialize the settings object
settings = Settings()

# 2. Run the directory check to make sure the app has a place to save files
settings.ensure_dirs()

