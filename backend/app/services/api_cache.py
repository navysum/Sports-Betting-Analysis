"""
Disk-based cache for football-data.org API responses.

Saves raw JSON responses to data/api_cache/ with a timestamp.
Callers check the cache before hitting the API — if the entry is
fresh enough, no network request is made.

TTL strategy (default 20 h):
  - Team histories, standings, finished matches: 20 h
    → data fetched at 01:00 AM stays valid until 21:00, then refreshes next morning
  - Today's scheduled fixtures: use a date-keyed entry; it auto-expires at midnight
"""
import json
import os
import time
from typing import Any, Optional

_CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "api_cache")
)

DEFAULT_TTL_HOURS = 20.0


def _path(key: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe = key.replace("/", "_").replace("?", "_").replace(":", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.json")


def get(key: str, ttl_hours: float = DEFAULT_TTL_HOURS) -> Optional[Any]:
    """Return cached data if fresh, else None."""
    path = _path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            entry = json.load(f)
        if (time.time() - entry["fetched_at"]) > ttl_hours * 3600:
            return None
        return entry["data"]
    except Exception:
        return None


def set(key: str, data: Any) -> None:
    """Save data to cache with current timestamp."""
    try:
        with open(_path(key), "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "data": data}, f)
    except Exception as e:
        print(f"[api_cache] Failed to save '{key}': {e}")


def age_hours(key: str) -> Optional[float]:
    """Return age of entry in hours, or None if not cached."""
    path = _path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            entry = json.load(f)
        return (time.time() - entry["fetched_at"]) / 3600
    except Exception:
        return None


def is_stale(key: str, ttl_hours: float = DEFAULT_TTL_HOURS) -> bool:
    """True if the cache entry is missing or older than ttl_hours."""
    age = age_hours(key)
    return age is None or age > ttl_hours


def any_stale(keys: list, ttl_hours: float = DEFAULT_TTL_HOURS) -> bool:
    """True if any key in the list is stale."""
    return any(is_stale(k, ttl_hours) for k in keys)
