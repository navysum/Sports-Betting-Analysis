from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.database import init_db
from app.api.matches import router as matches_router
from app.api.predictions import router as predictions_router
from ml.predict import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    load_model()
    yield


app = FastAPI(
    title="Sports Bet Analysis API",
    version="2.0.0",
    description=(
        "Autonomous football prediction API. "
        "Primary interface is Telegram — run run_bot.py for the full bot experience."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(matches_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Sports Bet Analysis API v2",
        "tip": "Run run_bot.py for the Telegram bot interface.",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/accuracy")
async def accuracy_endpoint():
    from app.services.evaluator import get_accuracy_stats
    return {
        "7d":  get_accuracy_stats(days=7),
        "30d": get_accuracy_stats(days=30),
        "all": get_accuracy_stats(),
    }
