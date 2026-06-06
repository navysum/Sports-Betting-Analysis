"""
DB-backed cache for football-data.org API responses.

Stores raw API responses in the api_cache table with a timestamp.
Falls back to the old disk cache (data/api_cache/*.json) transparently
so any existing cached files are still used until they expire.

Same public interface as before — get / set / age_hours / is_stale / any_stale —
so nothing in football_api.py or predictions.py needs to change.

TTL strategy (default 20 h):
  - Team histories, standings, finished matches: 20 h
  - Today's scheduled fixtures: date-keyed entry, auto-expires at midnight
"""
import json
import os
import time
import asyncio
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

DEFAULT_TTL_HOURS = 20.0

# Legacy disk cache dir — kept for read fallback only
_CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "api_cache")
)


# ---------------------------------------------------------------------------
# DB helpers (async, fire-and-forget for writes)
# ---------------------------------------------------------------------------

async def _db_get(key: str, ttl_hours: float) -> Optional[Any]:
    try:
        from app.database import AsyncSessionLocal
        from app.models.db_models import APICache
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(APICache).where(APICache.cache_key == key)
            )
            row = result.scalars().first()
            if row is None:
                return None
            if (time.time() - row.fetched_at) > ttl_hours * 3600:
                return None
            return row.data
    except Exception as exc:
        log.debug("api_cache DB get failed for '%s': %s", key, exc)
        return None


async def _db_set(key: str, data: Any) -> None:
    try:
        from app.database import AsyncSessionLocal
        from app.models.db_models import APICache
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(APICache).where(APICache.cache_key == key)
            )
            row = result.scalars().first()
            if row is None:
                row = APICache(cache_key=key)
                session.add(row)
            row.data       = data
            row.fetched_at = time.time()
            await session.commit()
    except Exception as exc:
        log.debug("api_cache DB set failed for '%s': %s", key, exc)


async def _db_age(key: str) -> Optional[float]:
    try:
        from app.database import AsyncSessionLocal
        from app.models.db_models import APICache
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(APICache).where(APICache.cache_key == key)
            )
            row = result.scalars().first()
            if row is None:
                return None
            return (time.time() - row.fetched_at) / 3600
    except Exception as exc:
        log.debug("api_cache DB age failed for '%s': %s", key, exc)
        return None


# ---------------------------------------------------------------------------
# Sync wrappers (keep the existing synchronous interface intact)
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context — use a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=5)
        else:
            return loop.run_until_complete(coro)
    except Exception as exc:
        log.debug("api_cache _run failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Disk fallback helpers (read-only, for existing cached files)
# ---------------------------------------------------------------------------

def _disk_path(key: str) -> str:
    safe = key.replace("/", "_").replace("?", "_").replace(":", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.json")


def _disk_get(key: str, ttl_hours: float) -> Optional[Any]:
    try:
        with open(_disk_path(key), encoding="utf-8") as f:
            entry = json.load(f)
        if (time.time() - entry["fetched_at"]) > ttl_hours * 3600:
            return None
        return entry["data"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def _disk_age(key: str) -> Optional[float]:
    try:
        with open(_disk_path(key), encoding="utf-8") as f:
            entry = json.load(f)
        return (time.time() - entry["fetched_at"]) / 3600
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Public API (same interface as before)
# ---------------------------------------------------------------------------

def get(key: str, ttl_hours: float = DEFAULT_TTL_HOURS) -> Optional[Any]:
    """Return cached data if fresh — checks DB first, then disk fallback."""
    # 1. Try DB
    data = _run(_db_get(key, ttl_hours))
    if data is not None:
        return data
    # 2. Fall back to old disk cache
    data = _disk_get(key, ttl_hours)
    if data is not None:
        # Migrate this entry into the DB so future reads hit DB
        _fire_set(key, data)
    return data


def set(key: str, data: Any) -> None:  # noqa: A001
    """Save data to DB cache (and keep a disk copy as backup)."""
    _fire_set(key, data)
    # Also write disk copy as a backup
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_disk_path(key), "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "data": data}, f)
    except Exception as e:
        log.debug("api_cache disk backup write failed '%s': %s", key, e)


def _fire_set(key: str, data: Any) -> None:
    """Schedule async DB write without blocking."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_db_set(key, data))
        else:
            loop.run_until_complete(_db_set(key, data))
    except Exception as exc:
        log.debug("api_cache _fire_set failed: %s", exc)


def age_hours(key: str) -> Optional[float]:
    """Return age of entry in hours — checks DB first, then disk."""
    age = _run(_db_age(key))
    if age is not None:
        return age
    return _disk_age(key)


def is_stale(key: str, ttl_hours: float = DEFAULT_TTL_HOURS) -> bool:
    """True if the cache entry is missing or older than ttl_hours."""
    age = age_hours(key)
    return age is None or age > ttl_hours


def any_stale(keys: list, ttl_hours: float = DEFAULT_TTL_HOURS) -> bool:
    """True if any key in the list is stale."""
    return any(is_stale(k, ttl_hours) for k in keys)
