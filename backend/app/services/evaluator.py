"""
Post-match evaluation engine.

Responsibilities:
  - Check finished matches against logged predictions in the ledger
  - Mark predictions as correct / incorrect
  - Compute rolling accuracy stats
  - Write a brief post-mortem for each settled prediction
  - Persist everything back to the JSON ledger and SQLite
"""
import json
import math
import os
from datetime import datetime, timedelta
from typing import Optional
import asyncio

from app.config import settings
from app.services.football_api import get_finished_matches, FDORG_COMPETITIONS

LEDGER_PATH = os.path.join(settings.data_dir, "predictions.json")


def _load_ledger() -> list[dict]:
    if not os.path.exists(LEDGER_PATH):
        return []
    try:
        with open(LEDGER_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_ledger(ledger: list[dict]) -> None:
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def append_prediction(entry: dict) -> None:
    """Add a new prediction to the ledger."""
    ledger = _load_ledger()
    # Avoid duplicates by match_id
    existing_ids = {e["match_id"] for e in ledger}
    if entry["match_id"] not in existing_ids:
        ledger.append(entry)
        _save_ledger(ledger)


def get_unsettled_predictions(days_back: int = 3) -> list[dict]:
    """Return predictions that haven't been evaluated yet."""
    ledger = _load_ledger()
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    return [
        e for e in ledger
        if e.get("actual") is None and e.get("date", "9999") >= cutoff
    ]


def settle_prediction(match_id: str, actual: dict) -> Optional[dict]:
    """
    Update a prediction entry with the actual result and compute correctness.
    actual = {result: "HOME/DRAW/AWAY", score: "2-1", over25: bool, btts: bool}
    Returns the updated entry.
    """
    ledger = _load_ledger()
    for entry in ledger:
        if entry["match_id"] == match_id:
            entry["actual"] = actual

            pred = entry.get("prediction", {})
            entry["correct"] = {
                "result": pred.get("result") == actual.get("result"),
                "over25": pred.get("over_2.5_predicted") == actual.get("over25"),
                "btts": pred.get("btts_predicted") == actual.get("btts"),
            }

            # Tag whether entry had a value bet flagged (for ROI tracking)
            entry["had_value_bet"] = bool(pred.get("value_bets"))

            # Auto post-mortem
            entry["post_mortem"] = _generate_post_mortem(entry, pred, actual)
            break

    _save_ledger(ledger)
    return next((e for e in ledger if e["match_id"] == match_id), None)


def _generate_post_mortem(entry: dict, pred: dict, actual: dict) -> str:
    """Generate a short text post-mortem for a settled prediction."""
    lines = []

    pred_result = pred.get("result", "?")
    act_result = actual.get("result", "?")
    score = actual.get("score", "?")

    if pred_result == act_result:
        lines.append(f"Result correct ({pred_result}) — final score {score}.")
    else:
        conf = pred.get("confidence", 0)
        lines.append(
            f"Result wrong: predicted {pred_result} (conf {conf:.0%}), actual {act_result} ({score})."
        )
        # Identify possible reason
        factors = entry.get("factors_used", [])
        if "home_advantage" in factors and act_result == "AWAY":
            lines.append("Home advantage may have been overweighted.")
        if "form" in factors and act_result == "DRAW":
            lines.append("Strong form trend didn't convert; draw more likely than form suggested.")

    if pred.get("over_2.5_predicted") is not None:
        o25_correct = pred.get("over_2.5_predicted") == actual.get("over25")
        lines.append(f"Over 2.5 {'correct' if o25_correct else 'incorrect'}.")

    if pred.get("btts_predicted") is not None:
        btts_correct = pred.get("btts_predicted") == actual.get("btts")
        lines.append(f"BTTS {'correct' if btts_correct else 'incorrect'}.")

    return " ".join(lines)


async def evaluate_recent_predictions() -> dict:
    """
    Pull recently finished matches from the API and settle any matching
    unsettled predictions. Returns a summary dict.
    """
    unsettled = get_unsettled_predictions(days_back=7)
    if not unsettled:
        return {"settled": 0, "message": "No unsettled predictions found."}

    settled_count = 0
    errors = []

    # Build a lookup of recently finished matches
    recent_results: dict[int, dict] = {}
    for code in list(FDORG_COMPETITIONS)[:5]:  # limit API calls
        try:
            matches = await get_finished_matches(code, limit=30)
            for m in matches:
                api_id = m.get("id")
                hg = m.get("score", {}).get("fullTime", {}).get("home")
                ag = m.get("score", {}).get("fullTime", {}).get("away")
                if api_id and hg is not None and ag is not None:
                    if hg > ag:
                        result = "HOME"
                    elif hg == ag:
                        result = "DRAW"
                    else:
                        result = "AWAY"
                    recent_results[api_id] = {
                        "result": result,
                        "score": f"{hg}-{ag}",
                        "over25": (hg + ag) > 2,
                        "btts": hg > 0 and ag > 0,
                    }
            await asyncio.sleep(7)
        except Exception as e:
            errors.append(str(e))

    for entry in unsettled:
        api_id = entry.get("api_match_id")
        if api_id and api_id in recent_results:
            settle_prediction(entry["match_id"], recent_results[api_id])
            settled_count += 1

    return {
        "settled": settled_count,
        "checked": len(unsettled),
        "errors": errors,
    }


def _log_loss(settled: list[dict]) -> Optional[float]:
    """
    Multiclass log-loss on settled predictions.
    Requires home_prob/draw_prob/away_prob stored in the ledger entry.
    Lower is better (perfect model = 0, random = ~1.10).
    """
    eps = 1e-7
    ll_sum = 0.0
    count = 0
    for e in settled:
        pred   = e.get("prediction", {})
        actual = e.get("actual", {})
        result = actual.get("result")
        if result not in ("HOME", "DRAW", "AWAY"):
            continue
        prob_map = {
            "HOME": pred.get("home_prob"),
            "DRAW": pred.get("draw_prob"),
            "AWAY": pred.get("away_prob"),
        }
        p = prob_map.get(result)
        if p is None:
            continue
        ll_sum += math.log(max(p, eps))
        count += 1
    return round(-ll_sum / count, 4) if count else None


def _brier_score(settled: list[dict]) -> Optional[float]:
    """
    Multiclass Brier score on settled predictions.
    Averaged over all 3 outcome classes. Range [0, 2]; lower is better.
    Random classifier = ~0.67, good model ≈ 0.18–0.22.
    """
    bs_sum = 0.0
    count  = 0
    outcome_idx = {"HOME": 0, "DRAW": 1, "AWAY": 2}
    for e in settled:
        pred   = e.get("prediction", {})
        actual = e.get("actual", {})
        result = actual.get("result")
        if result not in outcome_idx:
            continue
        probs = [
            pred.get("home_prob"),
            pred.get("draw_prob"),
            pred.get("away_prob"),
        ]
        if None in probs:
            continue
        actual_oh = [0.0, 0.0, 0.0]
        actual_oh[outcome_idx[result]] = 1.0
        bs = sum((probs[i] - actual_oh[i]) ** 2 for i in range(3)) / 3.0
        bs_sum += bs
        count  += 1
    return round(bs_sum / count, 4) if count else None


def get_accuracy_stats(days: Optional[int] = None) -> dict:
    """
    Compute accuracy, log-loss, and Brier score over settled predictions,
    optionally filtered to the last N days.

    Returns: {total, correct_result, result_accuracy, over25_accuracy,
              btts_accuracy, log_loss, brier_score, window_days}
    """
    ledger = _load_ledger()
    settled = [e for e in ledger if e.get("actual") is not None]

    if days is not None:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        settled = [e for e in settled if e.get("date", "0") >= cutoff]

    if not settled:
        return {
            "total":            0,
            "correct_result":   0,
            "result_accuracy":  0.0,
            "over25_accuracy":  None,
            "btts_accuracy":    None,
            "log_loss":         None,
            "brier_score":      None,
            "window_days":      days,
        }

    result_correct = [e for e in settled if e.get("correct", {}).get("result")]
    o25_settled    = [e for e in settled if e.get("correct", {}).get("over25") is not None]
    btts_settled   = [e for e in settled if e.get("correct", {}).get("btts")   is not None]

    return {
        "total":           len(settled),
        "correct_result":  len(result_correct),
        "result_accuracy": round(len(result_correct) / len(settled), 4),
        "over25_accuracy": (
            round(sum(1 for e in o25_settled if e["correct"]["over25"]) / len(o25_settled), 4)
            if o25_settled else None
        ),
        "btts_accuracy": (
            round(sum(1 for e in btts_settled if e["correct"]["btts"]) / len(btts_settled), 4)
            if btts_settled else None
        ),
        "log_loss":    _log_loss(settled),
        "brier_score": _brier_score(settled),
        "window_days": days,
    }


def get_accuracy_by_league(days: Optional[int] = None) -> dict:
    """
    Break down result accuracy by league.
    Returns dict of {league: {total, correct, accuracy}} sorted by volume.
    """
    ledger = _load_ledger()
    settled = [e for e in ledger if e.get("actual") is not None]

    if days:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        settled = [e for e in settled if e.get("date", "0") >= cutoff]

    leagues: dict[str, dict] = {}
    for e in settled:
        league = e.get("league", "Unknown")
        if league not in leagues:
            leagues[league] = {"total": 0, "correct": 0}
        leagues[league]["total"] += 1
        if e.get("correct", {}).get("result"):
            leagues[league]["correct"] += 1

    return {
        league: {
            **stats,
            "accuracy": round(stats["correct"] / stats["total"], 3) if stats["total"] else 0.0,
        }
        for league, stats in sorted(leagues.items(), key=lambda x: -x[1]["total"])
    }


def get_value_bet_roi(days: Optional[int] = None) -> dict:
    """
    Strike rate for matches where the model flagged a value bet.
    Returns {bets, wins, strike_rate} or {bets: 0} if no data.
    """
    ledger = _load_ledger()
    settled = [
        e for e in ledger
        if e.get("actual") is not None and e.get("had_value_bet")
    ]

    if days:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        settled = [e for e in settled if e.get("date", "0") >= cutoff]

    if not settled:
        return {"bets": 0, "wins": 0, "strike_rate": None}

    wins = sum(1 for e in settled if e.get("correct", {}).get("result"))
    return {
        "bets": len(settled),
        "wins": wins,
        "strike_rate": round(wins / len(settled), 3),
    }


def build_ledger_entry(
    match_id: str,
    api_match_id: Optional[int],
    date: str,
    league: str,
    home: str,
    away: str,
    prediction: dict,
    factors_used: list[str],
    key_factors: str,
) -> dict:
    """Construct a fully-formed ledger entry (without actual result yet)."""
    return {
        "match_id": match_id,
        "api_match_id": api_match_id,
        "date": date,
        "league": league,
        "home": home,
        "away": away,
        "prediction": prediction,
        "actual": None,
        "correct": None,
        "factors_used": factors_used,
        "key_factors": key_factors,
        "post_mortem": None,
        "logged_at": datetime.utcnow().isoformat(),
    }
