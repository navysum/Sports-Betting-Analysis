import asyncio  # Async I/O support for concurrent tasks
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
from app.api.admin import router as admin_router
from ml.predict import load_model
from app.services.football_api import build_team_cache

# Initialize the background scheduler to run periodic tasks
_scheduler = AsyncIOScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifecycle.
    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.
    """
    # 1. Initialize the database connection
    await init_db()
    
    # 2. Load the Machine Learning model into memory
    load_model()

    # 3. Build team search cache so /search works immediately after startup
    asyncio.create_task(build_team_cache())

    # 4. Kick off today's predictions immediately so they're ready when users arrive
    # This runs in the background so it doesn't block the app from starting
    asyncio.create_task(preload_today_predictions())

    # 4. Schedule a daily 6am UTC preload job
    # This ensures predictions are refreshed every morning automatically
    _scheduler.add_job(
        preload_today_predictions,
        CronTrigger(hour=6, minute=0, timezone="UTC"),
        id="daily_preload",
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
