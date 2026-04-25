import asyncio  # Async I/O support for concurrent tasks
import os
from fastapi import FastAPI  # Core FastAPI framework for building APIs
from fastapi.middleware.cors import CORSMiddleware  # Handles CORS (cross-origin requests)
from contextlib import asynccontextmanager  # Manages app lifecycle (startup/shutdown)

# Scheduler imports - handles background tasks on a timer
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Local application imports - importing logic from other parts of the project
from app.database import init_db
from app.api.matches import router as matches_router
from app.api.predictions import router as predictions_router, preload_today_predictions
from app.api.admin import router as admin_router, _run_retrain, _retrain_state
from app.api.ai import router as ai_router
from ml.predict import load_model, RESULT_MODEL_PATH
from app.services.football_api import build_team_cache

# Initialize the background scheduler to run periodic tasks
_scheduler = AsyncIOScheduler(timezone="UTC")


async def _settle_yesterday() -> None:
    """
    Nightly job (23:00 UTC): fetch yesterday's match results and update CLV log.

    For each prediction logged without closing odds, fetch the Pinnacle closing
    line and call clv_tracker.update_closing().  Skips entries that already have
    CLV filled in or are for today/future dates.
    """
    from datetime import date, timedelta
    from app.services.clv_tracker import _load_log, update_closing
    from app.services.football_api import get_odds

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    entries = _load_log()
    pending = [
        e for e in entries
        if e.get("clv") is None and e.get("date", "") <= yesterday
    ]

    if not pending:
        print("[settlement] No pending CLV entries to settle")
        return

    print(f"[settlement] Settling {len(pending)} CLV entries for dates up to {yesterday}")
    updated = 0
    for entry in pending:
        try:
            match_id = entry["id"]
            market   = entry["market"]
            odds     = await get_odds(match_id)
            if not odds:
                continue
            pinnacle = odds.get("pinnacle") or {}
            market_key = {
                "home":   "home",
                "draw":   "draw",
                "away":   "away",
                "over25": "over25",
            }.get(market)
            if market_key and pinnacle.get(market_key):
                update_closing(match_id, market, pinnacle[market_key])
                updated += 1
        except Exception as exc:
            print(f"[settlement] Error settling {entry.get('id')}: {exc}")

    print(f"[settlement] Done — updated={updated}, skipped={len(pending) - updated}")


async def _auto_train_then_preload() -> None:
    """
    Run when the service starts with no trained model (e.g. fresh Render deploy).
    Trains the model first, reloads it, then kicks off today's preload and
    the team cache build — exactly what normal startup does, just in order.
    """
    print("[startup] Auto-training model (this takes 5-10 min on first deploy)…")
    try:
        await _run_retrain()
        load_model()
        print("[startup] Auto-train complete — model loaded, starting preload")
    except Exception as e:
        print(f"[startup] Auto-train failed: {e} — preload will run but predictions may fail")
    asyncio.create_task(build_team_cache())
    asyncio.create_task(preload_today_predictions())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifecycle.
    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.
    """
    # 1. Initialize the database connection
    await init_db()
    
    # 2. Load the ML model — if missing, train first then load
    load_model()
    if not os.path.exists(RESULT_MODEL_PATH):
        print("[startup] No trained model found — running auto-train before preload…")
        # Train synchronously so preload only starts once the model is ready
        asyncio.create_task(_auto_train_then_preload())
    else:
        # 3. Build team search cache so /search works immediately after startup
        asyncio.create_task(build_team_cache())
        # 4. Kick off today's predictions immediately
        asyncio.create_task(preload_today_predictions())

    # 4. Schedule a daily 6am UTC preload job
    # This ensures predictions are refreshed every morning automatically
    _scheduler.add_job(
        preload_today_predictions,
        CronTrigger(hour=6, minute=0, timezone="UTC"),
        id="daily_preload",
        replace_existing=True,
    )

    # Nightly settlement: walk yesterday's predictions, fetch results, compute CLV
    _scheduler.add_job(
        _settle_yesterday,
        CronTrigger(hour=23, minute=0, timezone="UTC"),
        id="nightly_settlement",
        replace_existing=True,
    )

    _scheduler.start()

    yield # --- The app is now running and handling requests ---

    # 5. Clean up on shutdown: Stop the scheduler
    _scheduler.shutdown(wait=False)


# Create the FastAPI app instance with metadata and the lifespan manager
app = FastAPI(
    title="Sports Bet Analysis API",
    description="Backend API for Sports Betting Analysis with ML Predictions",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows your frontend (e.g. React/Next.js) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change this for production)
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Connect (include) the modular routers from the app/api folder
# Every route in these files will be prefixed with /api
app.include_router(matches_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(ai_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint to check if the API is online."""
    return {"status": "ok", "message": "Sports Bet Analysis API v2"}


@app.get("/health")
async def health():
    """Health check endpoint for monitoring tools."""
    return {"status": "healthy"}


@app.get("/accuracy")
async def accuracy_endpoint():
    """Fetches accuracy statistics for the ML model over different timeframes."""
    from app.services.evaluator import get_accuracy_stats
    return {
        "7d":  get_accuracy_stats(days=7),
        "30d": get_accuracy_stats(days=30),
        "all": get_accuracy_stats(),
    }
