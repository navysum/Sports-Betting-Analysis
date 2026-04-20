"""
Full evaluation report for the Sports Betting Analysis model.

Produces a structured report covering:
  - Summary (total bets, hit rate, ROI, drawdown, avg edge, avg CLV)
  - ROI by market
  - ROI by league
  - ROI by odds bucket
  - ROI by edge bucket
  - ROI by confidence bucket
  - ROI by fallback status
  - Calibration table (expected vs actual hit rate by probability decile)
  - CLV summary
  - Longest losing streak
  - Rolling 100-bet and 250-bet performance

Data sources:
  - backend/data/clv_log.json  — prediction + CLV records
  - backend/data/ai_decisions_log.json — AI decision records (if present)

Usage:
  python -m ml.evaluation_report
  or import generate_report() from another script / API endpoint.
"""
from __future__ import annotations
import json
import os
import math
from datetime import datetime
from typing import Optional

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CLV_LOG   = os.path.join(_DATA_DIR, "clv_log.json")
_AI_LOG    = os.path.join(_DATA_DIR, "ai_decisions_log.json")


def _load_json(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _parse_ts(iso: str) -> float:
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def _odds_band(odds: Optional[float]) -> str:
    if odds is None:
        return "unknown"
    if odds < 1.40:  return "1.20-1.39"
    if odds < 1.60:  return "1.40-1.59"
    if odds < 1.80:  return "1.60-1.79"
    if odds < 2.10:  return "1.80-2.09"
    if odds < 2.50:  return "2.10-2.49"
    return "2.50+"


def _edge_band(edge: Optional[float]) -> str:
    if edge is None or edge <= 0:
        return "0.00-0.02"
    if edge < 0.02:  return "0.00-0.02"
    if edge < 0.04:  return "0.02-0.04"
    if edge < 0.06:  return "0.04-0.06"
    if edge < 0.10:  return "0.06-0.10"
    return "0.10+"


def _conf_band(prob: Optional[float]) -> str:
    if prob is None:
        return "unknown"
    if prob < 0.55:  return "0.50-0.55"
    if prob < 0.60:  return "0.55-0.60"
    if prob < 0.65:  return "0.60-0.65"
    if prob < 0.70:  return "0.65-0.70"
    return "0.70+"


def _roi_stats(bets: list[dict]) -> dict:
    """Compute hit_rate, ROI, avg_edge, avg_clv, max_drawdown from settled bets."""
    if not bets:
        return {"count": 0}

    wins = sum(1 for b in bets if b.get("actual_outcome") is True)
    pnls = [b["pnl"] for b in bets if b.get("pnl") is not None]
    clvs = [b["clv"] for b in bets if b.get("clv") is not None]
    edges = [b.get("edge", 0) for b in bets if b.get("edge") is not None]

    # Max drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    return {
        "count":       len(bets),
        "wins":        wins,
        "hit_rate":    round(wins / len(bets), 3) if bets else None,
        "total_pnl":   round(sum(pnls), 2) if pnls else None,
        "roi":         round(sum(pnls) / len(pnls) * 100, 1) if pnls else None,
        "avg_edge":    round(sum(edges) / len(edges), 4) if edges else None,
        "avg_clv":     round(sum(clvs) / len(clvs), 4) if clvs else None,
        "max_drawdown": round(max_dd, 2),
    }


def _longest_losing_streak(bets: list[dict]) -> int:
    streak = 0
    best = 0
    for b in sorted(bets, key=lambda x: x.get("logged_at", "")):
        if b.get("actual_outcome") is False:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _calibration_table(bets: list[dict]) -> list[dict]:
    """
    Expected vs actual hit rate in probability deciles.
    Uses model_prob and actual_outcome.
    """
    deciles = [(i * 0.1, (i + 1) * 0.1) for i in range(5, 10)]  # 0.50–1.00
    rows = []
    for lo, hi in deciles:
        bucket = [b for b in bets if b.get("model_prob") is not None
                  and lo <= b["model_prob"] < hi
                  and b.get("actual_outcome") is not None]
        if not bucket:
            continue
        actual = sum(1 for b in bucket if b["actual_outcome"]) / len(bucket)
        expected = sum(b["model_prob"] for b in bucket) / len(bucket)
        rows.append({
            "bucket":   f"{lo:.0%}–{hi:.0%}",
            "count":    len(bucket),
            "expected": round(expected, 3),
            "actual":   round(actual, 3),
            "diff":     round(actual - expected, 3),
        })
    return rows


def _rolling_performance(bets: list[dict], window: int) -> list[dict]:
    """Rolling ROI over `window` settled bets."""
    settled = [b for b in bets if b.get("pnl") is not None]
    settled.sort(key=lambda x: x.get("logged_at", ""))
    rows = []
    for i in range(window - 1, len(settled)):
        chunk = settled[i - window + 1: i + 1]
        pnls = [b["pnl"] for b in chunk]
        roi = sum(pnls) / len(pnls) * 100
        rows.append({
            "bet_number": i + 1,
            "roi":        round(roi, 1),
        })
    return rows


def generate_report(days: int = 365) -> dict:
    """
    Generate the full evaluation report.

    Args:
        days: lookback window (default 365 = full year)

    Returns:
        Nested dict with all report sections.
    """
    import time
    cutoff = time.time() - days * 86400

    # Load AI decisions (richer: has model_prob, edge, market, fallback flags, outcome)
    ai_log = _load_json(_AI_LOG)
    clv_log = _load_json(_CLV_LOG)

    # Filter to window and settled records
    settled = [
        e for e in ai_log
        if e.get("actual_outcome") is not None
        and _parse_ts(e.get("logged_at", "")) >= cutoff
    ]

    # Enrich settled records with CLV data from clv_log
    clv_by_id = {}
    for c in clv_log:
        key = f"{c.get('id')}|{c.get('market')}"
        if c.get("clv") is not None:
            clv_by_id[key] = c["clv"]

    for s in settled:
        clv_key = f"{s.get('match_id')}|{s.get('market')}"
        if clv_key in clv_by_id:
            s["clv"] = clv_by_id[clv_key]

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _roi_stats(settled)
    summary["longest_losing_streak"] = _longest_losing_streak(settled)
    summary["total_predictions"] = len(ai_log)
    summary["predictions_with_outcome"] = len(settled)
    summary["days_window"] = days

    # ── By market ─────────────────────────────────────────────────────────────
    markets = {}
    for b in settled:
        m = b.get("market", "unknown")
        markets.setdefault(m, []).append(b)
    by_market = {m: _roi_stats(v) for m, v in markets.items()}

    # ── By league ─────────────────────────────────────────────────────────────
    leagues = {}
    for b in settled:
        lg = b.get("league") or b.get("competition_code", "unknown")
        leagues.setdefault(lg, []).append(b)
    by_league = {lg: _roi_stats(v) for lg, v in leagues.items()}

    # ── By odds bucket ─────────────────────────────────────────────────────────
    odds_buckets: dict = {}
    for b in settled:
        band = _odds_band(b.get("bookmaker_odds"))
        odds_buckets.setdefault(band, []).append(b)
    by_odds_bucket = {k: _roi_stats(v) for k, v in odds_buckets.items()}

    # ── By edge bucket ─────────────────────────────────────────────────────────
    edge_buckets: dict = {}
    for b in settled:
        band = _edge_band(b.get("edge"))
        edge_buckets.setdefault(band, []).append(b)
    by_edge_bucket = {k: _roi_stats(v) for k, v in edge_buckets.items()}

    # ── By confidence bucket ───────────────────────────────────────────────────
    conf_buckets: dict = {}
    for b in settled:
        band = _conf_band(b.get("model_prob"))
        conf_buckets.setdefault(band, []).append(b)
    by_conf_bucket = {k: _roi_stats(v) for k, v in conf_buckets.items()}

    # ── By fallback status ─────────────────────────────────────────────────────
    def _fallback_key(b: dict) -> str:
        flags = b.get("fallback_flags", {})
        parts = []
        if not flags.get("used_xg_fallback"):
            parts.append("real_xg")
        else:
            parts.append("fallback_xg")
        if not flags.get("used_dc_fallback"):
            parts.append("dc_found")
        else:
            parts.append("no_dc")
        return "+".join(parts)

    fallback_buckets: dict = {}
    for b in settled:
        key = _fallback_key(b)
        fallback_buckets.setdefault(key, []).append(b)
    by_fallback = {k: _roi_stats(v) for k, v in fallback_buckets.items()}

    # ── Calibration table ──────────────────────────────────────────────────────
    calibration = _calibration_table(settled)

    # ── CLV summary ────────────────────────────────────────────────────────────
    with_clv = [b for b in settled if b.get("clv") is not None]
    clvs = [b["clv"] for b in with_clv]
    clv_summary = {
        "count":          len(with_clv),
        "avg_clv":        round(sum(clvs) / len(clvs), 4) if clvs else None,
        "positive_clv_pct": round(sum(1 for c in clvs if c > 0) / len(clvs), 3) if clvs else None,
    }

    # ── Rolling performance ────────────────────────────────────────────────────
    rolling_100  = _rolling_performance(settled, 100)
    rolling_250  = _rolling_performance(settled, 250)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "summary":      summary,
        "by_market":    by_market,
        "by_league":    by_league,
        "by_odds_bucket":       by_odds_bucket,
        "by_edge_bucket":       by_edge_bucket,
        "by_confidence_bucket": by_conf_bucket,
        "by_fallback_status":   by_fallback,
        "calibration":  calibration,
        "clv_summary":  clv_summary,
        "rolling_100":  rolling_100[-20:] if rolling_100 else [],
        "rolling_250":  rolling_250[-20:] if rolling_250 else [],
    }


if __name__ == "__main__":
    import pprint
    report = generate_report()
    pprint.pprint(report, depth=3)
