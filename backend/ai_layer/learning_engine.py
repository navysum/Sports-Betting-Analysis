"""
Learning Engine — logs AI decisions and tracks performance over time.

Every AI recommendation is written to the ai_decisions table.
After settlement, actual outcomes are recorded so grades can be evaluated.

This enables:
  - Which AI grades actually perform best
  - Which warnings matter most
  - Which leagues are strongest
  - Which markets are traps
  - Where AI overrates edge
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Optional

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


def log_decision(
    match_id: str,
    match_info: dict,
    ai_result: dict,
) -> None:
    """
    Persist one AI decision to the append-only JSON log.

    Args:
        match_id:   Stable match identifier
        match_info: {home_team, away_team, league, competition_code, match_date}
        ai_result:  Output from recommendation_service.analyze_market()
    """
    entries = _load_log()
    packet = ai_result.get("packet", {})

    entry = {
        "id":             f"{match_id}|{ai_result.get('market_key', '')}",
        "logged_at":      datetime.utcnow().isoformat(),
        "match_id":       match_id,
        "match_date":     match_info.get("match_date", ""),
        "league":         match_info.get("league", ""),
        "competition_code": match_info.get("competition_code", ""),
        "home_team":      match_info.get("home_team", ""),
        "away_team":      match_info.get("away_team", ""),
        "market":         ai_result.get("market_key", ""),
        "market_label":   ai_result.get("market", ""),
        "recommendation": ai_result.get("recommendation", "PASS"),
        "grade":          ai_result.get("grade", "F"),
        "score":          ai_result.get("score", 0.0),
        "risk_level":     ai_result.get("risk_level", "HIGH"),
        "eligible":       ai_result.get("eligible", False),
        "model_prob":     packet.get("model_probability"),
        "edge":           packet.get("edge"),
        "bookmaker_odds": packet.get("bookmaker_odds"),
        "fallback_flags": packet.get("fallback_flags", {}),
        "reasoning":      ai_result.get("reasoning", []),
        "warnings":       ai_result.get("warnings", []),
        "stake_modifier": ai_result.get("stake_modifier", 0.0),
        # Post-settlement fields (filled later)
        "actual_outcome": None,
        "pnl":            None,
        "settled_at":     None,
    }

    # Deduplicate by id — update existing rather than append
    existing_ids = {e["id"]: i for i, e in enumerate(entries)}
    if entry["id"] in existing_ids:
        entries[existing_ids[entry["id"]]] = {**entries[existing_ids[entry["id"]]], **entry}
    else:
        entries.append(entry)

    _save_log(entries)


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
    """
    entries = _load_log()
    entry_id = f"{match_id}|{market}"
    for entry in entries:
        if entry.get("id") == entry_id:
            entry["actual_outcome"] = actual_outcome
            entry["pnl"] = pnl
            entry["settled_at"] = datetime.utcnow().isoformat()
            break
    _save_log(entries)


def get_performance_summary(days: int = 30) -> dict:
    """
    Summarise AI decision performance.

    Returns grade-level and market-level performance stats.
    """
    import time
    entries = _load_log()
    cutoff = time.time() - days * 86400

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
            "settled": 0,
            "by_grade": {},
            "by_market": {},
            "by_recommendation": {},
            "days": days,
        }

    def _stats(subset: list[dict]) -> dict:
        wins = sum(1 for e in subset if e.get("actual_outcome"))
        pnls = [e["pnl"] for e in subset if e.get("pnl") is not None]
        return {
            "count": len(subset),
            "win_rate": round(wins / len(subset), 3) if subset else None,
            "total_pnl": round(sum(pnls), 2) if pnls else None,
            "roi": round(sum(pnls) / len(pnls) * 100, 1) if pnls else None,
        }

    by_grade: dict = {}
    by_market: dict = {}
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
