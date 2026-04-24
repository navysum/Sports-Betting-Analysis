"""
Closing Line Value (CLV) tracker.

CLV is the single most important metric for a sharp bettor: if your model
consistently beats the Pinnacle closing line, you have a proven edge.

How it works:
  1. When a prediction is made, we record the model probability and the
     opening Pinnacle odds at that moment.
  2. At match time (or shortly after the market closes), we record the
     final Pinnacle closing odds.
  3. CLV = model_prob - (1 / pinnacle_closing_odds)
     Positive CLV = you found value. Negative CLV = market was right.

Long-term: if avg CLV > 0 across many predictions, the model has real edge.
If avg CLV ≤ 0, improve the model or stop betting.

Stored in data/clv_log.json (append-only, human-readable).
"""
import json
import os
import time
from datetime import datetime
from typing import Optional

CLV_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "clv_log.json",
)


def _load_log() -> list[dict]:
    try:
        with open(CLV_LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_log(entries: list[dict]):
    os.makedirs(os.path.dirname(CLV_LOG_PATH), exist_ok=True)
    with open(CLV_LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def log_prediction(
    match_id: str,
    match_date: str,
    home_team: str,
    away_team: str,
    competition: str,
    market: str,
    model_prob: float,
    opening_odds: Optional[dict],
    pinnacle_opening_odds: Optional[dict],
) -> None:
    """
    Record a prediction with opening odds at time of prediction.

    market: "home" | "draw" | "away" | "over25" | "btts" | "over35"
    opening_odds: best available odds across bookmakers
    pinnacle_opening_odds: Pinnacle-specific odds (the sharp reference)
    """
    entries = _load_log()

    def _implied(odds_dict: Optional[dict], key: str) -> Optional[float]:
        if not odds_dict or key not in odds_dict:
            return None
        try:
            price = odds_dict[key]
            return round(1.0 / price, 4) if price and price > 1 else None
        except Exception:
            return None

    # FIX #5: map market names to the correct odds dict keys.
    # Previously over25/btts/over35 were mapped to None, so every goals-market CLV
    # entry had implied_prob=None and was unusable. Now over25 maps to the "over25"
    # key that _extract_best_odds() now populates from the Pinnacle totals market.
    # btts and over35 remain None (not available from Pinnacle via The Odds API);
    # those entries are still logged but clearly show no implied probability.
    market_key = {
        "home":   "home",
        "draw":   "draw",
        "away":   "away",
        "over25": "over25",   # now populated from Pinnacle totals market
        "over35": None,       # not available via The Odds API
        "btts":   None,       # not available via The Odds API
    }.get(market, market)

    entry = {
        "id":                      match_id,
        "date":                    match_date,
        "home_team":               home_team,
        "away_team":               away_team,
        "competition":             competition,
        "market":                  market,
        "model_prob":              round(model_prob, 4),
        "opening_implied":         _implied(opening_odds, market_key) if market_key else None,
        "pinnacle_opening_implied":_implied(pinnacle_opening_odds, market_key) if market_key else None,
        "pinnacle_closing_implied":None,   # filled in by update_closing()
        "clv":                     None,   # filled in by update_closing()
        "logged_at":               datetime.utcnow().isoformat(),
    }

    entries.append(entry)
    _save_log(entries)


def update_closing(
    match_id: str,
    market: str,
    pinnacle_closing_odds: float,
) -> None:
    """
    Record the Pinnacle closing odds and compute CLV for a logged prediction.
    Call this after the market closes (before kick-off).
    """
    entries = _load_log()
    closing_implied = round(1.0 / pinnacle_closing_odds, 4) if pinnacle_closing_odds > 1 else None

    for entry in entries:
        if entry["id"] == match_id and entry["market"] == market and entry["clv"] is None:
            entry["pinnacle_closing_implied"] = closing_implied
            if closing_implied and entry["model_prob"]:
                # CLV = how much better our model prob is vs the closing implied
                entry["clv"] = round(entry["model_prob"] - closing_implied, 4)
            break

    _save_log(entries)


def get_clv_stats(days: int = 30) -> dict:
    """
    Return CLV performance summary.

    {
      total_predictions: int,
      predictions_with_clv: int,
      avg_clv: float,           # positive = consistent edge
      positive_clv_rate: float, # % of predictions beating closing line
      by_market: {market: {avg_clv, count, positive_rate}},
    }
    """
    entries = _load_log()

    cutoff = time.time() - days * 86400
    relevant = [
        e for e in entries
        if e.get("clv") is not None
        and _parse_ts(e.get("logged_at", "")) >= cutoff
    ]

    if not relevant:
        return {
            "total_predictions": len(entries),
            "predictions_with_clv": 0,
            "avg_clv": None,
            "positive_clv_rate": None,
            "by_market": {},
        }

    clvs = [e["clv"] for e in relevant]
    avg_clv = round(sum(clvs) / len(clvs), 4)
    positive_rate = round(sum(1 for c in clvs if c > 0) / len(clvs), 4)

    by_market: dict[str, dict] = {}
    for e in relevant:
        m = e["market"]
        by_market.setdefault(m, []).append(e["clv"])

    market_stats = {
        m: {
            "avg_clv":       round(sum(vs) / len(vs), 4),
            "count":         len(vs),
            "positive_rate": round(sum(1 for v in vs if v > 0) / len(vs), 4),
        }
        for m, vs in by_market.items()
    }

    return {
        "total_predictions":   len(entries),
        "predictions_with_clv": len(relevant),
        "avg_clv":             avg_clv,
        "positive_clv_rate":   positive_rate,
        "by_market":           market_stats,
        "days":                days,
    }


def get_clv_timeseries(days: int = 90) -> list[dict]:
    """
    Return daily CLV aggregates for the last N days, ordered oldest-first.

    Each item: { date, avg_clv, count, beat_close_rate, cumulative_avg }

    Suitable for rendering a rolling CLV bar/line chart in the frontend.
    """
    from datetime import timezone, timedelta

    entries = _load_log()
    cutoff  = datetime.now(timezone.utc) - timedelta(days=days)

    # Bucket entries by date
    by_date: dict[str, list[float]] = {}
    for e in entries:
        clv = e.get("clv")
        if clv is None:
            continue
        ts = _parse_ts(e.get("logged_at", ""))
        if ts < cutoff.timestamp():
            continue
        date = e.get("date") or e.get("logged_at", "")[:10]
        by_date.setdefault(date, []).append(clv)

    if not by_date:
        return []

    rows = []
    running_sum = 0.0
    running_n   = 0
    for date in sorted(by_date.keys()):
        clvs = by_date[date]
        day_avg = sum(clvs) / len(clvs)
        running_sum += sum(clvs)
        running_n   += len(clvs)
        rows.append({
            "date":            date,
            "avg_clv":         round(day_avg, 4),
            "count":           len(clvs),
            "beat_close_rate": round(sum(1 for c in clvs if c > 0) / len(clvs), 4),
            "cumulative_avg":  round(running_sum / running_n, 4),
        })
    return rows


def _parse_ts(iso: str) -> float:
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0
