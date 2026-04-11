from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # Football APIs
    football_data_api_key: str = ""
    api_football_key: str = ""  # RapidAPI free tier — 100 req/day

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/soccerbet.db"

    # Telegram
    telegram_token: str = ""
    telegram_chat_id: str = "-5002959525"   # Jarvis HQ group
    telegram_user_id: str = "5757363641"    # Your personal ID (admin commands)

    # Filesystem paths (relative to backend/)
    data_dir: str = "./data"
    models_dir: str = "./ml"
    csv_dir: str = "./data/csv"

    # Scheduling timezone
    timezone: str = "Europe/London"

    class Config:
        env_file = ".env"

    def ensure_dirs(self):
        for d in [self.data_dir, self.models_dir, self.csv_dir]:
            os.makedirs(d, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
