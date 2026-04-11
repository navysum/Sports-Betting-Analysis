"""
Scheduled tasks — wired into PTB's JobQueue (APScheduler under the hood).
All times are Europe/London (BST/GMT).

Schedule:
  02:00  Daily data scrape
  06:30  Generate & cache predictions
  07:00  Send morning briefing to Telegram
  Every 30 min on match days  — live score checks
  Every 60 min                — evaluate settled predictions
  Sunday 03:00                — weekly model retrain
  Sunday 09:00                — weekly performance report
"""
import asyncio
import datetime
import logging

import pytz
from telegram import Bot
from telegram.constants import ParseMode
from telegram.ext import Application

from app.config import settings
from app.services.evaluator import (
    evaluate_recent_predictions,
    get_accuracy_stats,
)
from app.services.scraper import bulk_download_historical
from app.services.football_api import get_all_today_matches, FDORG_COMPETITIONS
from bot.formatter import (
    format_daily_briefing,
    format_weekly_report,
)

log = logging.getLogger(__name__)
TZ = pytz.timezone(settings.timezone)

CHAT_ID = settings.telegram_chat_id


async def _send(bot: Bot, text: str) -> None:
    try:
        await bot.send_message(
            chat_id=CHAT_ID,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except Exception as e:
        log.warning(f"[scheduler] send failed: {e}")
        # Try plain text fallback
        try:
            import re
            clean = re.sub(r"[_*`\[\]()~>#+=|{}.!\\]", "", text)
            await bot.send_message(chat_id=CHAT_ID, text=clean)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Task implementations
# ──────────────────────────────────────────────────────────────────────────────

async def task_daily_scrape(context) -> None:
    """02:00 — Download fresh CSV data from Football-Data.co.uk."""
    log.info("[scheduler] Starting daily scrape…")
    try:
        await bulk_download_historical(seasons=["2425"])
        log.info("[scheduler] Daily scrape complete.")
    except Exception as e:
        log.error(f"[scheduler] Daily scrape failed: {e}")


async def task_generate_predictions(context) -> None:
    """06:30 — Generate predictions for today's matches and cache them."""
    log.info("[scheduler] Generating predictions for today…")
    try:
        from bot.commands import _batch_predictions_for_today
        predictions = await _batch_predictions_for_today("Today")
        # Store in bot_data for the morning briefing
        context.bot_data["today_predictions"] = predictions
        log.info(f"[scheduler] Generated {len(predictions)} predictions.")
    except Exception as e:
        log.error(f"[scheduler] Prediction generation failed: {e}")
        context.bot_data["today_predictions"] = []


async def task_morning_briefing(context) -> None:
    """07:00 — Send the daily briefing to Telegram."""
    log.info("[scheduler] Sending morning briefing…")
    predictions = context.bot_data.get("today_predictions")

    if predictions is None:
        # Generate now if not cached
        try:
            from bot.commands import _batch_predictions_for_today
            predictions = await _batch_predictions_for_today("Today")
        except Exception as e:
            log.error(f"[scheduler] Late prediction fetch failed: {e}")
            predictions = []

    date_str = datetime.datetime.now(TZ).strftime("%A %-d %B %Y")
    text = format_daily_briefing(predictions, label="Today", date_str=date_str)
    await _send(context.bot, text)


async def task_live_scores(context) -> None:
    """Every 30 min on match days — check live scores and send goal alerts."""
    try:
        from app.services.football_api import get_live_matches

        live_any = False
        for code in list(FDORG_COMPETITIONS)[:5]:
            try:
                matches = await get_live_matches(code)
                for m in matches:
                    live_any = True
                    home = m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name", "?")
                    away = m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name", "?")
                    hg = m.get("score", {}).get("fullTime", {}).get("home") or m.get("score", {}).get("halfTime", {}).get("home", "?")
                    ag = m.get("score", {}).get("fullTime", {}).get("away") or m.get("score", {}).get("halfTime", {}).get("away", "?")
                    minute = m.get("minute") or "?"
                    status = m.get("status", "")

                    # Only alert on score changes (deduplicate via bot_data)
                    score_key = f"{m.get('id')}_{hg}_{ag}"
                    seen = context.bot_data.setdefault("seen_scores", set())
                    if score_key not in seen:
                        seen.add(score_key)
                        await _send(
                            context.bot,
                            f"⚽ *{home} {hg}\\-{ag} {away}* \\(`{minute}'`\\)",
                        )
                await asyncio.sleep(7)
            except Exception:
                pass

        if not live_any:
            log.debug("[scheduler] No live matches.")

    except Exception as e:
        log.error(f"[scheduler] Live score check failed: {e}")


async def task_evaluate_predictions(context) -> None:
    """Every 60 min — settle any finished predictions in the ledger."""
    try:
        result = await evaluate_recent_predictions()
        if result["settled"] > 0:
            log.info(f"[scheduler] Settled {result['settled']} predictions.")
    except Exception as e:
        log.error(f"[scheduler] Evaluation failed: {e}")


async def task_weekly_retrain(context) -> None:
    """Sunday 03:00 — Retrain all models on accumulated data."""
    log.info("[scheduler] Starting weekly retrain…")
    try:
        from ml.train import main as train_main
        await train_main()
        from ml.train import get_last_training_summary
        summary = get_last_training_summary()
        r_acc = summary.get("result_model", {}).get("accuracy_mean", 0)
        await _send(
            context.bot,
            f"✅ *Weekly retrain complete*\n"
            f"  Samples: {summary.get('samples', '?')} \\| Result model: {r_acc:.1%}",
        )
        log.info("[scheduler] Weekly retrain done.")
    except Exception as e:
        log.error(f"[scheduler] Weekly retrain failed: {e}")
        await _send(context.bot, f"❌ Weekly retrain failed: {e}")


async def task_weekly_report(context) -> None:
    """Sunday 09:00 — Send the weekly performance report + model health check."""
    log.info("[scheduler] Sending weekly report…")
    try:
        import json, os
        from bot.formatter import format_weekly_report
        from ml.train import get_last_training_summary

        bets_path = os.path.join(settings.data_dir, "bets.json")
        bets = json.load(open(bets_path)) if os.path.exists(bets_path) else []

        acc7 = get_accuracy_stats(days=7)
        summary = get_last_training_summary()
        text = format_weekly_report(acc7, bets, summary)
        await _send(context.bot, text)

        # ── Model health check ────────────────────────────────────────────────
        # Alert if 7-day accuracy drops below 35% on ≥10 settled predictions
        # (35% = barely above random for 1X2; anything below suggests model failure)
        _ACCURACY_FLOOR = 0.35
        _MIN_SAMPLE     = 10
        if (
            acc7.get("total", 0) >= _MIN_SAMPLE
            and acc7.get("result_accuracy", 1.0) < _ACCURACY_FLOOR
        ):
            acc_pct = acc7["result_accuracy"] * 100
            total   = acc7["total"]
            log.warning(
                f"[scheduler] Model health alert: {acc_pct:.1f}% accuracy "
                f"on {total} predictions — triggering emergency retrain."
            )
            alert = (
                f"⚠️ *Model Health Alert*\n"
                f"7\\-day accuracy: {acc_pct:.1f}% on {total} predictions\n"
                f"Below {int(_ACCURACY_FLOOR * 100)}% floor — "
                f"initiating emergency retrain…"
            )
            await _send(context.bot, alert)
            await task_weekly_retrain(context)

    except Exception as e:
        log.error(f"[scheduler] Weekly report failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────

def register_jobs(app: Application) -> None:
    """Register all scheduled jobs with PTB's JobQueue."""
    jq = app.job_queue

    # Daily scrape at 02:00 BST
    jq.run_daily(
        task_daily_scrape,
        time=datetime.time(2, 0, tzinfo=TZ),
        name="daily_scrape",
    )

    # Generate predictions at 06:30 BST
    jq.run_daily(
        task_generate_predictions,
        time=datetime.time(6, 30, tzinfo=TZ),
        name="generate_predictions",
    )

    # Morning briefing at 07:00 BST
    jq.run_daily(
        task_morning_briefing,
        time=datetime.time(7, 0, tzinfo=TZ),
        name="morning_briefing",
    )

    # Live score checks every 30 min
    jq.run_repeating(
        task_live_scores,
        interval=datetime.timedelta(minutes=30),
        first=datetime.timedelta(minutes=5),
        name="live_scores",
    )

    # Evaluate predictions every 60 min
    jq.run_repeating(
        task_evaluate_predictions,
        interval=datetime.timedelta(hours=1),
        first=datetime.timedelta(minutes=15),
        name="evaluate_predictions",
    )

    # Weekly retrain — Sunday 03:00 BST
    jq.run_daily(
        task_weekly_retrain,
        time=datetime.time(3, 0, tzinfo=TZ),
        days=(6,),   # 0=Mon … 6=Sun
        name="weekly_retrain",
    )

    # Weekly report — Sunday 09:00 BST
    jq.run_daily(
        task_weekly_report,
        time=datetime.time(9, 0, tzinfo=TZ),
        days=(6,),
        name="weekly_report",
    )

    log.info("[scheduler] All jobs registered.")
