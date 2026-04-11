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
import os
from app.services.scraper import bulk_download_historical
from app.services.football_api import get_all_today_matches, FDORG_COMPETITIONS, build_team_cache
from bot.formatter import (
    format_daily_briefing,
    format_weekly_report,
    format_saved_tips_report,
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


async def task_nightly_tips_report(context) -> None:
    """23:00 — Send result of user-saved tips, then delete them from the list."""
    log.info("[scheduler] Sending nightly tips report…")
    try:
        import json
        saved_path  = os.path.join(settings.data_dir, "saved_tips.json")
        ledger_path = os.path.join(settings.data_dir, "predictions.json")

        if not os.path.exists(saved_path):
            return

        with open(saved_path) as f:
            all_tips = json.load(f)

        if not all_tips:
            return

        # Load ledger to look up settled results
        ledger: list[dict] = []
        if os.path.exists(ledger_path):
            with open(ledger_path) as f:
                ledger = json.load(f)

        ledger_by_id = {e["match_id"]: e for e in ledger}

        # Enrich each tip with its result (settled or still pending)
        enriched = []
        for tip in all_tips:
            mid   = tip.get("match_id")
            entry = ledger_by_id.get(mid, {})
            actual = entry.get("actual")
            enriched.append({
                **tip,
                "actual":  actual,
                "correct": entry.get("correct", {}),
                "settled": actual is not None,
            })

        text = format_saved_tips_report(enriched)
        await _send(context.bot, text)

        # Delete all tips that have been settled — unsettled ones stay for tomorrow
        remaining = [
            tip for tip, enr in zip(all_tips, enriched)
            if not enr["settled"]
        ]
        with open(saved_path, "w") as f:
            json.dump(remaining, f, indent=2)

        settled_count = len(enriched) - len(remaining)
        log.info(
            f"[scheduler] Nightly tips report sent. "
            f"{settled_count} deleted, {len(remaining)} carried forward."
        )

    except Exception as e:
        log.error(f"[scheduler] Nightly tips report failed: {e}")


async def task_weekly_report(context) -> None:
    """Sunday 09:00 — Send the weekly performance report."""
    log.info("[scheduler] Sending weekly report…")
    try:
        import json
        from bot.formatter import format_weekly_report
        from ml.train import get_last_training_summary

        bets_path = os.path.join(settings.data_dir, "bets.json")
        bets = json.load(open(bets_path)) if os.path.exists(bets_path) else []

        acc7 = get_accuracy_stats(days=7)
        summary = get_last_training_summary()
        text = format_weekly_report(acc7, bets, summary)
        await _send(context.bot, text)

        # Alert if 7-day accuracy is below 30% (with enough data)
        if acc7.get("total", 0) >= 10 and acc7.get("result_accuracy", 1.0) < 0.30:
            acc_pct = f"{acc7['result_accuracy']:.0%}"
            await _send(
                context.bot,
                f"⚠️ *Model alert*: 7\\-day accuracy is {acc_pct} \\(below 30%\\)\\. "
                "Consider running /retrain\\.",
            )
    except Exception as e:
        log.error(f"[scheduler] Weekly report failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────

async def task_build_team_cache(context) -> None:
    """Run once 15s after startup — builds the team name lookup cache."""
    log.info("[scheduler] Building team lookup cache…")
    try:
        await build_team_cache()
        log.info("[scheduler] Team cache ready.")
    except Exception as e:
        log.error(f"[scheduler] Team cache build failed: {e}")


def register_jobs(app: Application) -> None:
    """Register all scheduled jobs with PTB's JobQueue."""
    jq = app.job_queue

    # Build team cache 15s after startup (non-blocking)
    jq.run_once(task_build_team_cache, when=15, name="build_team_cache")

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

    # Nightly tips report — 23:00 BST every day
    jq.run_daily(
        task_nightly_tips_report,
        time=datetime.time(23, 0, tzinfo=TZ),
        name="nightly_tips_report",
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
