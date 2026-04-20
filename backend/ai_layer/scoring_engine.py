"""
Scoring Engine — weighted composite score 0–10 for an eligible bet.

Formula (from File 2, Section 6):
  score = edge_score    * 0.35
        + conf_score    * 0.25
        + hist_score    * 0.20
        + clv_score     * 0.10
        + quality_score * 0.10

Each component is normalised to [0, 10] before weighting.
"""
from __future__ import annotations
from typing import Optional


# Weights (must sum to 1.0)
W_EDGE    = 0.35
W_CONF    = 0.25
W_HIST    = 0.20
W_CLV     = 0.10
W_QUALITY = 0.10


def _edge_score(edge: float, market: str) -> float:
    """
    Normalise edge to 0–10.
    5% edge  → ~3.5/10 (decent)
    10% edge → ~7.0/10 (strong)
    15%+ edge → 10/10 (exceptional)
    """
    if edge <= 0:
        return 0.0
    # Logarithmic scale: score = 10 × min(1, ln(1 + edge/0.02) / ln(1 + 0.15/0.02))
    import math
    max_ref = 0.15
    score = 10.0 * min(1.0, math.log(1 + edge / 0.02) / math.log(1 + max_ref / 0.02))
    return round(score, 2)


def _confidence_score(model_prob: float, market: str) -> float:
    """
    Normalise model probability to 0–10.
    Min useful prob (market-specific) maps to ~3; 1.0 → 10.
    """
    min_probs = {
        "home": 0.50, "draw": 0.38, "away": 0.48,
        "over25": 0.55, "btts": 0.53, "over35": 0.58,
    }
    min_p = min_probs.get(market, 0.50)
    if model_prob <= min_p:
        return 0.0
    score = (model_prob - min_p) / (1.0 - min_p) * 10.0
    return round(min(score, 10.0), 2)


def _historical_score(roi: Optional[float], bets: int) -> float:
    """
    Normalise historical segment ROI to 0–10.
    ROI ≤ -5%  → 0
    ROI = 0%   → 4 (neutral)
    ROI = +5%  → 6
    ROI = +10% → 8
    ROI = +15% → 10
    Weight by sample size (fewer bets → regression toward neutral 4).
    """
    if roi is None or bets < 5:
        return 4.0  # neutral when unknown

    base = 4.0 + roi * 0.4   # +1 per 2.5% ROI above 0; -1 per 2.5% below 0
    score = max(0.0, min(10.0, base))

    # Dampen toward neutral (4) when sample is small
    if bets < 30:
        dampen = min(1.0, bets / 30.0)
        score = 4.0 + (score - 4.0) * dampen

    return round(score, 2)


def _clv_score(clv_avg: Optional[float], beat_rate: Optional[float]) -> float:
    """
    Normalise CLV performance to 0–10.
    avg_clv > 0 and beat_rate > 0.55 → high scores.
    No CLV data → neutral 5.
    """
    if clv_avg is None:
        return 5.0

    # avg CLV component (−0.05 → 0, 0 → 5, +0.05 → 10)
    clv_component = 5.0 + clv_avg * 100.0   # +1 per 1% CLV
    clv_component = max(0.0, min(10.0, clv_component))

    # beat_rate component
    if beat_rate is not None:
        br_component = (beat_rate - 0.5) * 20.0 + 5.0   # 50% → 5, 75% → 10
        br_component = max(0.0, min(10.0, br_component))
        return round((clv_component + br_component) / 2, 2)

    return round(clv_component, 2)


def _quality_score(data_quality: float, warnings: list[str]) -> float:
    """
    Normalise data quality to 0–10.
    quality=1.0, 0 warnings → 10
    quality=0.0, many warnings → 0
    """
    base = data_quality * 10.0
    penalty = len(warnings) * 1.0
    return round(max(0.0, min(10.0, base - penalty)), 2)


def compute_score(packet: dict, warnings: list[str]) -> dict:
    """
    Compute the weighted composite score and all component scores.

    Returns:
        {
          "total": float,           # 0–10 composite
          "grade": str,             # A / B / C / D / F
          "components": {...},      # breakdown
        }
    """
    market = packet.get("market", "")

    e_score = _edge_score(packet.get("edge", 0.0), market)
    c_score = _confidence_score(packet.get("model_probability", 0.0), market)
    h_score = _historical_score(
        packet.get("historical_segment_roi"),
        packet.get("historical_segment_bets", 0),
    )
    clv_s = _clv_score(
        packet.get("clv_history"),
        packet.get("clv_beat_rate"),
    )
    q_score = _quality_score(
        packet.get("data_quality_score", 1.0),
        warnings,
    )

    total = round(
        e_score    * W_EDGE
        + c_score  * W_CONF
        + h_score  * W_HIST
        + clv_s    * W_CLV
        + q_score  * W_QUALITY,
        2,
    )

    if total >= 8.5:
        grade = "A"
    elif total >= 7.0:
        grade = "B"
    elif total >= 5.5:
        grade = "C"
    elif total >= 4.0:
        grade = "D"
    else:
        grade = "F"

    return {
        "total": total,
        "grade": grade,
        "components": {
            "edge_score":    e_score,
            "conf_score":    c_score,
            "hist_score":    h_score,
            "clv_score":     clv_s,
            "quality_score": q_score,
        },
    }
