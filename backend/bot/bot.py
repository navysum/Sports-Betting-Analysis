"""
Telegram bot entry point.

Sets up the Application, registers command handlers, and starts polling.
Run from the backend/ directory:
    python -m bot.bot
or via the top-level wrapper:
    python run_bot.py
"""
import logging
import sys

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from app.config import settings
from ml.predict import load_model
from bot.commands import (
    cmd_today,
    cmd_tomorrow,
    cmd_analyse,
    cmd_a,
    cmd_form,
    cmd_h2h,
    cmd_injuries,
    cmd_standings,
    cmd_bet,
    cmd_settle,
    cmd_bets,
    cmd_pnl,
    cmd_accuracy,
    cmd_leaguestats,
    cmd_backtest,
    cmd_retrain,
    cmd_leagues,
    cmd_saved,
    cmd_help,
    handle_save_prediction,
)
from bot.scheduler import register_jobs

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def build_app() -> Application:
    if not settings.telegram_token:
        raise RuntimeError(
            "TELEGRAM_TOKEN not set. Add it to your .env file and restart."
        )

    # Load ML models once at startup
    load_model()

    app = (
        Application.builder()
        .token(settings.telegram_token)
        .build()
    )

    # Register command handlers
    handlers = [
        ("today",        cmd_today),
        ("tomorrow",     cmd_tomorrow),
        ("analyse",      cmd_analyse),
        ("a",            cmd_a),
        ("form",         cmd_form),
        ("h2h",          cmd_h2h),
        ("injuries",     cmd_injuries),
        ("standings",    cmd_standings),
        ("bet",          cmd_bet),
        ("settle",       cmd_settle),
        ("bets",         cmd_bets),
        ("pnl",          cmd_pnl),
        ("accuracy",     cmd_accuracy),
        ("leaguestats",  cmd_leaguestats),
        ("backtest",     cmd_backtest),
        ("retrain",      cmd_retrain),
        ("leagues",      cmd_leagues),
        ("saved",        cmd_saved),
        ("help",         cmd_help),
        ("start",        cmd_help),
    ]
    for name, handler in handlers:
        app.add_handler(CommandHandler(name, handler))

    # Reply handler — user replies to any bot message with "save prediction"
    app.add_handler(MessageHandler(
        filters.REPLY & filters.TEXT & filters.Regex(r"(?i)save\s+prediction"),
        handle_save_prediction,
    ))

    # Register scheduled jobs
    register_jobs(app)

    log.info("Bot built and handlers registered.")
    return app


def run() -> None:
    app = build_app()
    log.info(f"Starting bot (polling) — Chat: {settings.telegram_chat_id}")
    app.run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    run()
