"""
Admin endpoints: model retraining, status, and last training summary.
"""
import asyncio
import traceback
from datetime import datetime, timezone
from fastapi import APIRouter

router = APIRouter(prefix="/admin", tags=["admin"])

# ── Retrain state ─────────────────────────────────────────────────────────────
_retrain_state = {
    "status": "idle",          # idle | running | done | failed
    "started_at": None,
    "finished_at": None,
    "log": [],
    "error": None,
    "summary": None,
}
_retrain_lock = asyncio.Lock()


def _log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _retrain_state["log"].append(line)
    print(line)


async def _run_retrain():
    """Full training pipeline — runs in background."""
    global _retrain_state
    _retrain_state.update({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "log": [],
        "error": None,
        "summary": None,
    })

    try:
        _log("Starting retrain…")

        # Step 1 — load cache
        _log("Loading accumulated data cache…")
        from ml.train import _load_cache, _merge, _save_cache, train_all, CACHE_PATH
        X_cache, yr_cache, yg_cache, yb_cache = _load_cache()
        _log(f"Cache: {len(X_cache)} samples")

        # Step 2 — FDCO CSV data
        _log("Downloading Football-Data.co.uk CSVs…")
        from ml.fdco_trainer import build_fdco_training_data, download_all_csvs
        await download_all_csvs()
        X_fdco, yr_fdco, yg_fdco, yb_fdco, odds_rows_fdco = build_fdco_training_data()
        _log(f"FDCO: {len(X_fdco)} samples")

        # Step 3 — API data
        _log("Fetching API training data (rate-limited, this takes a few minutes)…")
        from ml.train import fetch_api_training_data
        X_api, yr_api, yg_api, yb_api = await fetch_api_training_data()
        _log(f"API: {len(X_api)} samples")

        # Step 4 — merge + save cache
        X, y_result, y_goals, y_btts = _merge(
            (X_cache, yr_cache, yg_cache, yb_cache),
            (X_fdco,  yr_fdco,  yg_fdco,  yb_fdco),
            (X_api,   yr_api,   yg_api,   yb_api),
        )
        _log(f"Merged: {len(X)} unique samples")

        if len(X) < 50:
            raise RuntimeError(f"Not enough training data ({len(X)} samples). Check API key and CSV files.")

        _save_cache(X, y_result, y_goals, y_btts)

        # Step 5 — train
        _log("Training models…")
        summary = train_all(X, y_result, y_goals, y_btts, odds_rows=odds_rows_fdco)
        _log("Models trained and saved.")

        # Step 6 — reload models in-memory
        _log("Reloading models into memory…")
        from ml.predict import load_model
        load_model()

        # Also reset today's cache so it re-runs with new models
        from app.api.predictions import _today_cache, preload_today_predictions
        _today_cache.clear()
        asyncio.create_task(preload_today_predictions())
        _log("Today's prediction cache cleared — recomputing with new models.")

        _retrain_state.update({
            "status": "done",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        })
        _log("Retrain complete.")

    except Exception as e:
        tb = traceback.format_exc()
        _log(f"ERROR: {e}")
        _retrain_state.update({
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        })
        print(tb)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/retrain")
async def trigger_retrain():
    """Start a full model retrain in the background (idempotent while running)."""
    async with _retrain_lock:
        if _retrain_state["status"] == "running":
            return {"message": "Retrain already in progress", "status": "running"}
        asyncio.create_task(_run_retrain())
    return {"message": "Retrain started", "status": "running"}


@router.get("/retrain/status")
async def retrain_status():
    """Poll this to track retrain progress."""
    from ml.train import get_last_training_summary
    state = dict(_retrain_state)
    # Include last training summary from log file (persists across restarts)
    if state["status"] == "idle":
        state["last_training"] = get_last_training_summary()
    # Trim log to last 50 lines for response
    state["log"] = state["log"][-50:]
    return state
