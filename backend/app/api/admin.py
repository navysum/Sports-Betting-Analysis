"""
Admin endpoints: model retraining, status, last training summary, and data cache refresh.
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
        X_cache, yr_cache, yg_cache, yb_cache, yo_cache, dates_cache = _load_cache()
        _log(f"Cache: {len(X_cache)} samples")

        # Step 2 — FDCO CSV data
        _log("Downloading Football-Data.co.uk CSVs…")
        from ml.fdco_trainer import build_fdco_training_data, download_all_csvs
        await download_all_csvs()
        X_fdco, yr_fdco, yg_fdco, yb_fdco, yo_fdco, odds_rows_fdco, dates_fdco = build_fdco_training_data()
        _log(f"FDCO: {len(X_fdco)} samples")

        # Step 3 — API data
        _log("Fetching API training data (rate-limited, this takes a few minutes)…")
        from ml.train import fetch_api_training_data
        X_api, yr_api, yg_api, yb_api, yo_api, dates_api = await fetch_api_training_data()
        _log(f"API: {len(X_api)} samples")

        # Step 4 — merge + save cache
        X, y_result, y_goals, y_btts, y_over35, dates_merged = _merge(
            (X_cache, yr_cache, yg_cache, yb_cache, yo_cache),
            (X_fdco,  yr_fdco,  yg_fdco,  yb_fdco,  yo_fdco),
            (X_api,   yr_api,   yg_api,   yb_api,   yo_api),
            dates_tuple=(dates_cache, dates_fdco, dates_api),
        )
        _log(f"Merged: {len(X)} unique samples")

        if len(X) < 50:
            raise RuntimeError(f"Not enough training data ({len(X)} samples). Check API key and CSV files.")

        _save_cache(X, y_result, y_goals, y_btts, y_over35, dates_merged)

        # Step 5 — train
        _log("Training models…")
        summary = train_all(X, y_result, y_goals, y_btts, y_over35, odds_rows=odds_rows_fdco)
        _log("Models trained and saved.")

        # Step 6 — reload models in-memory
        _log("Reloading models into memory…")
        from ml.predict import load_model
        load_model()

        # Also reset today's cache so it re-runs with new models.
        # Reset _preload_running too — an old stalled preload would block a new one.
        import app.api.predictions as _pred_mod
        _pred_mod._today_cache.clear()
        _pred_mod._preload_running = False
        asyncio.create_task(_pred_mod.preload_today_predictions())
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


# ── Cache refresh ─────────────────────────────────────────────────────────────

_refresh_lock = asyncio.Lock()
_refresh_state = {"status": "idle", "started_at": None, "finished_at": None, "fetched": 0}


async def _run_cache_refresh():
    """
    Pre-warm the API response cache for all leagues.
    Fetches standings + finished matches per league, then team histories
    for every team playing today. Subsequent prediction preloads and
    retrains skip the API entirely if this ran within the last 20 h.
    """
    global _refresh_state
    _refresh_state.update({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "fetched": 0,
    })
    fetched = 0
    try:
        from app.services.football_api import (
            get_standings, get_finished_matches, get_team_matches,
            get_all_today_matches, FDORG_COMPETITIONS,
        )

        # 1. Standings + finished matches for every league
        for comp in FDORG_COMPETITIONS:
            try:
                await get_standings(comp)
                fetched += 1
                await get_finished_matches(comp, limit=150)
                fetched += 1
                print(f"[cache-refresh] {comp}: standings + finished matches cached")
            except Exception as e:
                print(f"[cache-refresh] {comp} error: {e}")

        # 2. Today's fixtures — collect unique team IDs
        today_matches = await get_all_today_matches()
        team_ids: set[int] = set()
        for m in today_matches:
            h = m.get("homeTeam", {}).get("id")
            a = m.get("awayTeam", {}).get("id")
            if h:
                team_ids.add(h)
            if a:
                team_ids.add(a)

        # 3. Pre-fetch team histories for today's teams
        print(f"[cache-refresh] Pre-fetching histories for {len(team_ids)} teams playing today…")
        for team_id in team_ids:
            try:
                await get_team_matches(team_id, limit=25)
                fetched += 1
            except Exception as e:
                print(f"[cache-refresh] team {team_id} error: {e}")

        _refresh_state.update({
            "status": "done",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "fetched": fetched,
        })
        print(f"[cache-refresh] Complete — {fetched} entries cached.")

    except Exception as e:
        _refresh_state.update({
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "fetched": fetched,
        })
        print(f"[cache-refresh] Failed: {e}")


@router.post("/refresh-cache")
async def refresh_cache():
    """
    Pre-warm the on-disk API cache for all leagues and today's teams.
    Idempotent while running. Called by the GitHub Actions cron at 01:00 AM UTC.
    After this runs, prediction preloads and retrains skip live API calls.
    """
    async with _refresh_lock:
        if _refresh_state["status"] == "running":
            return {"message": "Cache refresh already in progress", "status": "running"}
        asyncio.create_task(_run_cache_refresh())
    return {"message": "Cache refresh started", "status": "running"}


@router.get("/refresh-cache/status")
async def refresh_cache_status():
    """Check the status of the last cache refresh."""
    return _refresh_state
