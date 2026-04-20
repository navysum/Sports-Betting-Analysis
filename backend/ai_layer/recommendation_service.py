"""
Recommendation Service — orchestrates packet_builder → rules_engine → scoring_engine.

Returns a structured AI decision for one or more markets.

Output format:
{
  "recommendation": "BET",          # STRONG BET / BET / SMALL BET / WATCHLIST / PASS / AVOID
  "grade": "B",                     # A / B / C / D / F
  "score": 7.4,                     # 0–10
  "risk_level": "LOW",
  "market": "Over 2.5 Goals",
  "reasoning": ["..."],
  "warnings": ["..."],
  "stake_modifier": 1.0,            # multiply base Kelly stake by this
  "eligible": True,
  "block_reason": "",
  "components": {...},              # score breakdown
}
"""
from __future__ import annotations
from typing import Optional

from ai_layer.packet_builder import build_packet, build_packets_all_markets
from ai_layer.rules_engine import apply_hard_rules, classify_risk
from ai_layer.scoring_engine import compute_score


def _recommendation_from_score(score: float) -> str:
    if score >= 8.5:
        return "STRONG BET"
    if score >= 7.0:
        return "BET"
    if score >= 6.0:
        return "SMALL BET"
    if score >= 5.0:
        return "WATCHLIST"
    return "PASS"


def _stake_modifier(score: float, risk_level: str) -> float:
    """Adjust Kelly stake based on score and risk."""
    base = 1.0
    if score >= 8.5:
        base = 1.5
    elif score >= 7.0:
        base = 1.0
    elif score >= 6.0:
        base = 0.75
    else:
        base = 0.5

    if risk_level == "HIGH":
        base *= 0.5
    elif risk_level == "MEDIUM":
        base *= 0.75

    return round(base, 2)


def _build_reasoning(packet: dict, score_breakdown: dict, recommendation: str) -> list[str]:
    """Generate human-readable reasoning strings."""
    reasons = []
    market = packet.get("market_label", packet.get("market", ""))
    edge = packet.get("edge", 0.0)
    model_prob = packet.get("model_probability", 0.0)
    flags = packet.get("fallback_flags", {})

    if edge >= 0.10:
        reasons.append(f"Strong edge: model shows {edge:.1%} over fair odds")
    elif edge >= 0.05:
        reasons.append(f"Solid edge: {edge:.1%} over fair implied probability")

    if model_prob >= 0.70:
        reasons.append(f"High model confidence: {model_prob:.0%}")
    elif model_prob >= 0.60:
        reasons.append(f"Good model confidence: {model_prob:.0%}")

    if not any(flags.values()):
        reasons.append("Full data quality: no fallbacks used")
    elif not flags.get("used_xg_fallback") and not flags.get("used_dc_fallback"):
        reasons.append("Core model data complete (xG + DC available)")

    hist_roi = packet.get("historical_segment_roi")
    hist_bets = packet.get("historical_segment_bets", 0)
    if hist_roi is not None and hist_roi > 0 and hist_bets >= 20:
        reasons.append(f"Positive historical ROI in this segment: {hist_roi:+.1f}% ({hist_bets} bets)")

    clv = packet.get("clv_history")
    clv_rate = packet.get("clv_beat_rate")
    if clv is not None and clv > 0:
        reasons.append(f"Model consistently beats the closing line (avg CLV {clv:+.3f})")
    if clv_rate is not None and clv_rate >= 0.55:
        reasons.append(f"Beats closing line {clv_rate:.0%} of the time")

    if packet.get("dc_available"):
        reasons.append("Dixon-Coles score model confirms fixture")

    if packet.get("league_model_used"):
        reasons.append("League-specific model active")

    if not reasons:
        reasons.append("Model detected edge above minimum threshold")

    return reasons


def analyze_market(
    prediction: dict,
    match_info: dict,
    market: str,
    historical_segment_roi: Optional[float] = None,
    historical_segment_bets: int = 0,
    clv_history: Optional[float] = None,
    clv_beat_rate: Optional[float] = None,
) -> dict:
    """
    Run the full AI analysis pipeline for a single market.

    Returns a structured decision dict.
    """
    packet = build_packet(
        prediction=prediction,
        match_info=match_info,
        market=market,
        historical_segment_roi=historical_segment_roi,
        historical_segment_bets=historical_segment_bets,
        clv_history=clv_history,
        clv_beat_rate=clv_beat_rate,
        injury_notes=prediction.get("adjustments", []),
    )

    eligible, block_reason, warnings = apply_hard_rules(packet)

    if not eligible:
        return {
            "recommendation": "PASS",
            "grade":          "F",
            "score":          0.0,
            "risk_level":     "HIGH",
            "market":         packet["market_label"],
            "market_key":     market,
            "reasoning":      [],
            "warnings":       [block_reason],
            "stake_modifier": 0.0,
            "eligible":       False,
            "block_reason":   block_reason,
            "components":     {},
            "packet":         packet,
        }

    risk_level = classify_risk(packet, warnings)
    score_result = compute_score(packet, warnings)
    total_score = score_result["total"]
    grade = score_result["grade"]
    recommendation = _recommendation_from_score(total_score)
    stake_mod = _stake_modifier(total_score, risk_level)
    reasoning = _build_reasoning(packet, score_result, recommendation)

    return {
        "recommendation": recommendation,
        "grade":          grade,
        "score":          total_score,
        "risk_level":     risk_level,
        "market":         packet["market_label"],
        "market_key":     market,
        "reasoning":      reasoning,
        "warnings":       warnings,
        "stake_modifier": stake_mod,
        "eligible":       True,
        "block_reason":   "",
        "components":     score_result["components"],
        "packet":         packet,
    }


def analyze_all_markets(
    prediction: dict,
    match_info: dict,
    clv_stats_by_market: Optional[dict] = None,
    historical_roi_by_segment: Optional[dict] = None,
) -> dict:
    """
    Analyze all markets with edge > 0.
    Returns best recommendation plus all market decisions.

    Returns:
        {
          "best_recommendation": {...},   # highest-scoring eligible market
          "all_markets": [...],           # all analyzed markets
          "match": str,
          "league": str,
          "has_value": bool,
        }
    """
    markets = ["home", "draw", "away", "over25", "btts", "over35"]
    edges = prediction.get("edges", {})
    clv_stats = clv_stats_by_market or {}
    hist_roi = historical_roi_by_segment or {}

    results = []
    for market in markets:
        if edges.get(market, 0) <= 0:
            continue

        comp = match_info.get("competition_code", "")
        seg_key = f"{comp}|{market}"

        result = analyze_market(
            prediction=prediction,
            match_info=match_info,
            market=market,
            historical_segment_roi=hist_roi.get(seg_key),
            historical_segment_bets=0,
            clv_history=clv_stats.get(market, {}).get("avg_clv"),
            clv_beat_rate=clv_stats.get(market, {}).get("positive_rate"),
        )
        results.append(result)

    eligible = [r for r in results if r["eligible"]]
    best = max(eligible, key=lambda r: r["score"]) if eligible else None

    return {
        "best_recommendation": best,
        "all_markets":         results,
        "match":               f"{match_info.get('home_team', '')} vs {match_info.get('away_team', '')}",
        "league":              match_info.get("league", ""),
        "competition_code":    match_info.get("competition_code", ""),
        "match_date":          match_info.get("match_date", ""),
        "has_value":           bool(eligible),
    }
