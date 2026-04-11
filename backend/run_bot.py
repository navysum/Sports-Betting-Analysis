"""
Entry point for the Sports Bet Analysis Telegram bot.

Usage (from the backend/ directory):
    python run_bot.py

Or via systemd on the VPS (see deployment notes below):
    ExecStart=/root/bots/trading/venv/bin/python /root/bots/trading/backend/run_bot.py

Requires:
    - .env file with TELEGRAM_TOKEN and FOOTBALL_DATA_API_KEY
    - ML models trained (run: python -m ml.train)
"""
import sys
import os

# Ensure backend/ is on the path when invoked from the project root
sys.path.insert(0, os.path.dirname(__file__))

from bot.bot import run

if __name__ == "__main__":
    run()
