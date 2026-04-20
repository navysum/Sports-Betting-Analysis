"""
Rules Engine — hard no-bet filters and eligibility gates.

Applies before the scoring engine. If any hard rule fails the result is PASS or AVOID
immediately, without computing a score. This separates "show prediction" from
"show as bet recommendation" as required by File 1, Section 4.4.

Rules are additive — the first failing HARD rule returns immediately.
SOFT rules log warnings but do not block.
"""
from __future__ import annotations
from typing import Optional


# Market-specific minimum confidence thresholds
_MIN_CONF = {
    "home":   0.55,
    "draw":   0.42,  # draws are hard, lower bar
    "away":   0.52,
    "over25": 0.60,
    "btts":   0.58,
    "over35": 0.62,
}

# Market-specific minimum edge thresholds
_MIN_EDGE = {
    "home":   0.06,
    "draw":   0.08,  # draws need higher edge to compensate for variance
    "away":   0.06,
    "over25": 0.05,
    "btts":   0.05,
    "over35": 0.06,
}

# Acceptable odds bands per market
_ODDS_BANDS = {
    "home":   (1.20, 3.50),
    "draw":   (2.50, 4.50),
    "away":   (1.30, 4.50),
    "over25": (1.40, 2.50),
    "btts":   (1.40, 2.20),
    "over35": (1.60, 3.20),
}


def apply_hard_rules(packet: dict) -> tuple[bool, str, list[str]]:
    """
    Apply hard eligibility rules.

    Returns:
        (eligible: bool, block_reason: str | "", warnings: list[str])

    Hard rules that block immediately:
        - no bookmaker odds available
        - approximate devigging only (no two-sided exact devig)
        - odds outside acceptable band
        - edge below minimum for this market
        - confidence below minimum for this market
        - DC fallback AND xG fallback simultaneously (too much uncertainty)
    """
    market = packet.get("market", "")
    edge = packet.get("edge", 0.0)
    model_prob = packet.get("model_probability", 0.0)
    book_odds = packet.get("bookmaker_odds")
    flags = packet.get("fallback_flags", {})
    warnings = []

    # Hard rule 1: no odds = can't evaluate value
    if not book_odds or book_odds <= 1.0:
        return False, "No bookmaker odds available", warnings

    # Hard rule 2: odds outside band
    band = _ODDS_BANDS.get(market, (1.0, 99.0))
    if not (band[0] <= book_odds <= band[1]):
        return False, f"Odds {book_odds:.2f} outside acceptable band {band[0]:.2f}–{band[1]:.2f}", warnings

    # Hard rule 3: edge below minimum
    min_edge = _MIN_EDGE.get(market, 0.05)
    if edge < min_edge:
        return False, f"Edge {edge:.1%} below minimum {min_edge:.1%} for {market}", warnings

    # Hard rule 4: confidence below minimum
    min_conf = _MIN_CONF.get(market, 0.55)
    if model_prob < min_conf:
        return False, f"Model prob {model_prob:.1%} below minimum {min_conf:.1%} for {market}", warnings

    # Hard rule 5: both DC and xG unavailable = too uncertain for 1X2 markets
    dc_fallback = flags.get("used_dc_fallback", False)
    xg_fallback = flags.get("used_xg_fallback", False)
    if market in ("home", "draw", "away") and dc_fallback and xg_fallback:
        return False, "Both DC and xG unavailable — 1X2 prediction too uncertain", warnings

    # ── Soft warnings (don't block, but lower score) ──────────────────────────
    if flags.get("used_approx_devig", False):
        warnings.append("Approximate devigging only — exact two-sided odds not available")

    if flags.get("used_global_model", False):
        warnings.append("Using global model — no league-specific model trained yet")

    if xg_fallback:
        warnings.append("xG derived from over-2.5 proxy, not real data")

    if dc_fallback:
        warnings.append("Dixon-Coles team not found — result model only")

    home_inj = packet.get("home_injuries", [])
    away_inj = packet.get("away_injuries", [])
    if len(home_inj) >= 3:
        warnings.append(f"{packet.get('home_team', 'Home team')} has {len(home_inj)} injury/suspensions")
    if len(away_inj) >= 3:
        warnings.append(f"{packet.get('away_team', 'Away team')} has {len(away_inj)} injury/suspensions")

    seg_bets = packet.get("historical_segment_bets", 0)
    if seg_bets < 20:
        warnings.append(f"Low historical sample in this segment ({seg_bets} bets)")

    seg_roi = packet.get("historical_segment_roi")
    if seg_roi is not None and seg_roi < -3.0:
        warnings.append(f"Historical segment ROI is negative ({seg_roi:.1f}%)")

    clv_beat = packet.get("clv_beat_rate")
    if clv_beat is not None and clv_beat < 0.45:
        warnings.append(f"CLV beat-rate below 50% ({clv_beat:.0%}) in this segment")

    return True, "", warnings


def classify_risk(packet: dict, warnings: list[str]) -> str:
    """Classify bet risk: LOW / MEDIUM / HIGH."""
    flags = packet.get("fallback_flags", {})
    n_fallbacks = sum([
        flags.get("used_xg_fallback", False),
        flags.get("used_dc_fallback", False),
        flags.get("used_global_model", False),
        flags.get("used_approx_devig", False),
    ])
    n_warnings = len(warnings)
    total_inj = len(packet.get("home_injuries", [])) + len(packet.get("away_injuries", []))

    risk_score = n_fallbacks + n_warnings * 0.5 + total_inj * 0.3
    if risk_score <= 1.0:
        return "LOW"
    if risk_score <= 3.0:
        return "MEDIUM"
    return "HIGH"
