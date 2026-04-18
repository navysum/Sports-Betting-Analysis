"""
Admin endpoints: model retraining, status, last training summary, and data cache refresh.
"""
import asyncio
import time
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


# ── CLV closing-line snapshot ──────────────────────────────────────────────────

@router.post("/clv-snapshot")
async def clv_snapshot():
    """
    Capture a closing-line snapshot for today's pending CLV entries.

    Should be called ~90 minutes before typical kickoff times so the Pinnacle
    odds captured are as close to true closing prices as possible.
    Called by the GitHub Actions cron at 11:30 UTC daily.

    For each CLV log entry from today (or yesterday) that still has no
    pinnacle_closing_implied, fetches current Pinnacle odds and records them.
    Returns counts of entries updated vs skipped.
    """
    asyncio.create_task(_run_clv_snapshot())
    return {"message": "CLV snapshot started", "status": "running"}


async def _run_clv_snapshot():
    """Background task: update CLV closing odds for pending log entries."""
    try:
        from app.services.clv_tracker import _load_log, update_closing
        from app.services.odds_api import find_pinnacle_odds, SPORT_MAP
        from app.services.football_api import SUPPORTED_COMPETITIONS

        entries = _load_log()
        # Current month prefix — only update entries from the last ~60 days
        cutoff_month = datetime.now(timezone.utc).strftime("%Y-%m")

        # Build reverse map: competition display name → competition code
        name_to_code = {v: k for k, v in SUPPORTED_COMPETITIONS.items()}

        market_to_key = {"home": "home", "draw": "draw", "away": "away", "over25": "over25"}

        pending = [
            e for e in entries
            if e.get("pinnacle_closing_implied") is None
            and e.get("date", "")[:7] >= cutoff_month[:7]  # same or later month
        ]

        print(f"[clv-snapshot] {len(pending)} pending CLV entries to update")
        updated = skipped = 0

        # Cache Pinnacle odds per (home, away, comp) to avoid redundant API calls
        seen_pairs: dict[str, object] = {}
        for entry in pending:
            comp_name = entry.get("competition", "")
            comp_code = name_to_code.get(comp_name) or comp_name  # fallback: use as-is
            pair_key = f"{entry.get('home_team')}|{entry.get('away_team')}|{comp_code}"

            if pair_key not in seen_pairs:
                if comp_code not in SPORT_MAP:
                    seen_pairs[pair_key] = None
                else:
                    try:
                        seen_pairs[pair_key] = await find_pinnacle_odds(
                            entry.get("home_team", ""),
                            entry.get("away_team", ""),
                            comp_code,
                        )
                    except Exception as exc:
                        print(f"[clv-snapshot] odds fetch error {pair_key}: {exc}")
                        seen_pairs[pair_key] = None

            pin_odds = seen_pairs[pair_key]
            if not pin_odds:
                skipped += 1
                continue

            mkt = entry.get("market")
            odds_key = market_to_key.get(mkt)
            if not odds_key or odds_key not in pin_odds:
                skipped += 1
                continue

            closing_price = pin_odds[odds_key]
            if closing_price and closing_price > 1.0:
                update_closing(entry["id"], mkt, closing_price)
                updated += 1
            else:
                skipped += 1

        print(f"[clv-snapshot] Done — updated={updated}, skipped={skipped}")

    except Exception as exc:
        print(f"[clv-snapshot] Failed: {exc}")


@router.get("/clv-stats")
async def clv_stats(days: int = 30):
    """Return CLV performance summary for the last N days."""
    from app.services.clv_tracker import get_clv_stats
    return get_clv_stats(days=days)


# ── Backtest ───────────────────────────────────────────────────────────────────

# Cache backtest result for 6 hours — it's CPU-intensive (predicts on ~11k rows)
_backtest_cache: dict = {}
_backtest_cache_ts: float = 0.0
_BACKTEST_TTL: float = 6 * 3600


@router.get("/backtest")
async def get_backtest(min_edge: float = 0.05, holdout: float = 0.30):
    """
    Run (or return cached) backtest results on FDCO historical data.

    Uses the last `holdout` fraction of chronologically sorted data as the
    test set. Evaluates the full production pipeline: calibrated XGBoost +
    Dixon-Coles blend + devigged odds — exactly what the live system uses.

    Results are cached for 6 hours to avoid hammering CPU on every page load.
    Trigger a fresh run by changing min_edge or holdout parameters.

    Returns staking ROI for result market (flat/value/kelly) and O/U 2.5
    (flat/value), plus Brier calibration scores for all four markets.
    """
    global _backtest_cache, _backtest_cache_ts

    cache_key = f"{min_edge}_{holdout}"
    if (cache_key == _backtest_cache.get("_key")
            and time.time() - _backtest_cache_ts < _BACKTEST_TTL):
        return _backtest_cache

    try:
        from ml.backtest import run_backtest
        # Run synchronously — takes ~5–15 s depending on data size
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_backtest(holdout_fraction=holdout, min_edge=min_edge),
        )
        result["_key"] = cache_key
        result["_cached_at"] = datetime.now(timezone.utc).isoformat()
        _backtest_cache = result
        _backtest_cache_ts = time.time()
        return result
    except Exception as e:
        return {"error": str(e)}
