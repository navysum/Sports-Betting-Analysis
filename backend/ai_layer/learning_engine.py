"""
Learning Engine — logs AI decisions and tracks performance over time.

Every AI recommendation is written to BOTH ai_decisions_log.json AND
the ai_decisions table in SQLite.

After settlement, actual outcomes are recorded so grades can be evaluated.

This enables:
  - Which AI grades actually perform best
  - Which warnings matter most
  - Which leagues are strongest
  - Which markets are traps
  - Where AI overrates edge

Dual-write strategy:
  - JSON is the primary / source of truth (unchanged behaviour)
  - DB write is background / best-effort — failure only logs a warning
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "ai_decisions_log.json",
)


def _load_log() -> list[dict]:
    try:
        with open(_LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_log(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    with open(_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _upsert_ai_decision_db(entry: dict) -> None:
    """
    Upsert one AI decision into the ai_decisions table.
    Silently swallows any DB error so the JSON path is never affected.
    """
    try:
        from app.database import AsyncSessionLocal
        from app.models.db_models import AIDecision
        from sqlalchemy import select

        row_id = entry.get("id", "")           # "{match_id}|{market}"
        parts  = row_id.split("|", 1)
        match_id = parts[0] if parts else row_id
        market   = parts[1] if len(parts) > 1 else entry.get("market", "")

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AIDecision).where(
                    AIDecision.match_id == match_id,
                    AIDecision.market   == market,
                )
            )
            row = result.scalars().first()

            if row is None:
                row = AIDecision(match_id=match_id, market=market)
                session.add(row)

            row.match_date      = entry.get("match_date")
            row.league          = entry.get("league")
            row.home_team       = entry.get("home_team")
            row.away_team       = entry.get("away_team")
            row.recommendation  = entry.get("recommendation")
            row.grade           = entry.get("grade")
            row.score           = entry.get("score")
            row.risk_level      = entry.get("risk_level")
            row.model_prob      = entry.get("model_prob")
            row.edge            = entry.get("edge")
            row.bookmaker_odds  = entry.get("bookmaker_odds")
            row.confidence      = entry.get("model_prob")   # best proxy available
            row.bet_eligible    = bool(entry.get("eligible", False))
            row.reasoning       = entry.get("reasoning")
            row.warnings        = entry.get("warnings")
            row.stake_modifier  = entry.get("stake_modifier", 1.0)

            # Fallback flags
            flags = entry.get("fallback_flags", {})
            row.used_xg_fallback     = bool(flags.get("xg_fallback"))
            row.used_dc_fallback     = bool(flags.get("dc_fallback"))
            row.used_global_model    = bool(flags.get("global_model"))
            row.used_approx_devig    = bool(flags.get("approx_devig"))

            # Settlement fields (populated later via update_outcome)
            if entry.get("actual_outcome") is not None:
                row.actual_outcome = entry["actual_outcome"]
            if entry.get("pnl") is not None:
                row.pnl = entry["pnl"]
            if entry.get("settled_at"):
                try:
                    row.settled_at = datetime.fromisoformat(entry["settled_at"])
                except Exception:
                    pass

            await session.commit()

    except Exception as exc:
        log.warning("DB upsert failed for AI decision %s (JSON still saved): %s",
                    entry.get("id"), exc)


def _fire_db_write(entry: dict) -> None:
    """Schedule a DB upsert without blocking the caller."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_upsert_ai_decision_db(entry))
        else:
            loop.run_until_complete(_upsert_ai_decision_db(entry))
    except Exception as exc:
        log.warning("Could not schedule AI decision DB write: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_decision(
    match_id: str,
    match_info: dict,
    ai_result: dict,
) -> None:
    """
    Persist one AI decision to the JSON log and mirror to SQLite.

    Args:
        match_id:   Stable match identifier
        match_info: {home_team, away_team, league, competition_code, match_date}
        ai_result:  Output from recommendation_service.analyze_market()
    """
    entries = _load_log()
    packet  = ai_result.get("packet", {})

    entry = {
        "id":               f"{match_id}|{ai_result.get('market_key', '')}",
        "logged_at":        datetime.utcnow().isoformat(),
        "match_id":         match_id,
        "match_date":       match_info.get("match_date", ""),
        "league":           match_info.get("league", ""),
        "competition_code": match_info.get("competition_code", ""),
        "home_team":        match_info.get("home_team", ""),
        "away_team":        match_info.get("away_team", ""),
        "market":           ai_result.get("market_key", ""),
        "market_label":     ai_result.get("market", ""),
        "recommendation":   ai_result.get("recommendation", "PASS"),
        "grade":            ai_result.get("grade", "F"),
        "score":            ai_result.get("score", 0.0),
        "risk_level":       ai_result.get("risk_level", "HIGH"),
        "eligible":         ai_result.get("eligible", False),
        "model_prob":       packet.get("model_probability"),
        "edge":             packet.get("edge"),
        "bookmaker_odds":   packet.get("bookmaker_odds"),
        "fallback_flags":   packet.get("fallback_flags", {}),
        "reasoning":        ai_result.get("reasoning", []),
        "warnings":         ai_result.get("warnings", []),
        "stake_modifier":   ai_result.get("stake_modifier", 0.0),
        # Post-settlement fields (filled later)
        "actual_outcome":   None,
        "pnl":              None,
        "settled_at":       None,
    }

    # Deduplicate by id — update existing rather than append
    existing_ids = {e["id"]: i for i, e in enumerate(entries)}
    if entry["id"] in existing_ids:
        entries[existing_ids[entry["id"]]] = {**entries[existing_ids[entry["id"]]], **entry}
    else:
        entries.append(entry)

    _save_log(entries)           # JSON — primary
    _fire_db_write(entry)        # DB  — secondary


def log_decisions_batch(
    match_id: str,
    match_info: dict,
    ai_results: list[dict],
) -> None:
    """Log all market decisions for one match."""
    for result in ai_results:
        try:
            log_decision(match_id, match_info, result)
        except Exception:
            pass


def update_outcome(
    match_id: str,
    market: str,
    actual_outcome: bool,
    pnl: Optional[float] = None,
) -> None:
    """
    Record actual outcome after settlement.
    actual_outcome=True means the bet won.
    Writes to JSON then mirrors to DB.
    """
    entries  = _load_log()
    entry_id = f"{match_id}|{market}"
    updated  = None

    for entry in entries:
        if entry.get("id") == entry_id:
            entry["actual_outcome"] = actual_outcome
            entry["pnl"]            = pnl
            entry["settled_at"]     = datetime.utcnow().isoformat()
            updated = entry
            break

    _save_log(entries)           # JSON — primary

    if updated:
        _fire_db_write(updated)  # DB  — secondary


# ---------------------------------------------------------------------------
# Read helpers (read from JSON — unchanged)
# ---------------------------------------------------------------------------

def get_performance_summary(days: int = 30) -> dict:
    """Summarise AI decision performance by grade / market / recommendation."""
    import time
    entries = _load_log()
    cutoff  = time.time() - days * 86400

    def _parse_ts(iso: str) -> float:
        try:
            from datetime import timezone
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return 0.0

    settled = [
        e for e in entries
        if e.get("actual_outcome") is not None
        and _parse_ts(e.get("logged_at", "")) >= cutoff
    ]

    if not settled:
        return {
            "total_decisions": len(entries),
            "settled":         0,
            "by_grade":        {},
            "by_market":       {},
            "by_recommendation": {},
            "days":            days,
        }

    def _stats(subset: list[dict]) -> dict:
        wins = sum(1 for e in subset if e.get("actual_outcome"))
        pnls = [e["pnl"] for e in subset if e.get("pnl") is not None]
        return {
            "count":     len(subset),
            "win_rate":  round(wins / len(subset), 3) if subset else None,
            "total_pnl": round(sum(pnls), 2) if pnls else None,
            "roi":       round(sum(pnls) / len(pnls) * 100, 1) if pnls else None,
        }

    by_grade:          dict = {}
    by_market:         dict = {}
    by_recommendation: dict = {}

    for e in settled:
        by_grade.setdefault(e.get("grade", "?"), []).append(e)
        by_market.setdefault(e.get("market", "?"), []).append(e)
        by_recommendation.setdefault(e.get("recommendation", "?"), []).append(e)

    return {
        "total_decisions":   len(entries),
        "settled":           len(settled),
        "by_grade":          {g: _stats(v) for g, v in by_grade.items()},
        "by_market":         {m: _stats(v) for m, v in by_market.items()},
        "by_recommendation": {r: _stats(v) for r, v in by_recommendation.items()},
        "days":              days,
    }


def get_recent_decisions(limit: int = 50, eligible_only: bool = True) -> list[dict]:
    """Return most recent AI decisions."""
    entries = _load_log()
    if eligible_only:
        entries = [e for e in entries if e.get("eligible")]
    entries.sort(key=lambda e: e.get("logged_at", ""), reverse=True)
    return entries[:limit]
