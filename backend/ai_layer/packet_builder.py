"""
Build structured prediction packets for the AI Decision Layer.

Takes raw output from ml/predict.py + contextual data and returns
a fully structured dict that rules_engine and scoring_engine can consume.
"""
from __future__ import annotations
from typing import Optional


def build_packet(
    prediction: dict,
    match_info: dict,
    market: str,
    historical_segment_roi: Optional[float] = None,
    historical_segment_bets: int = 0,
    clv_history: Optional[float] = None,
    clv_beat_rate: Optional[float] = None,
    injury_notes: Optional[list] = None,
) -> dict:
    """
    Build a structured analysis packet for a specific market.

    Args:
        prediction:   Output dict from ml/predict.py
        match_info:   {home_team, away_team, league, competition_code,
                       match_date, kickoff_utc}
        market:       "home" | "draw" | "away" | "over25" | "btts" | "over35"
        historical_segment_roi: ROI % from historical bets in this league+market+odds segment
        historical_segment_bets: number of historical bets in this segment
        clv_history:  average CLV for similar past bets
        clv_beat_rate: fraction of past bets that beat the closing line
        injury_notes: list of injury strings from prediction_service

    Returns:
        Structured packet dict consumed by rules_engine / scoring_engine.
    """
    market_prob_key = {
        "home":   "home_win_prob",
        "draw":   "draw_prob",
        "away":   "away_win_prob",
        "over25": "over25_prob",
        "btts":   "btts_prob",
        "over35": "over35_prob",
    }

    market_label = {
        "home":   "Home Win",
        "draw":   "Draw",
        "away":   "Away Win",
        "over25": "Over 2.5 Goals",
        "btts":   "BTTS Yes",
        "over35": "Over 3.5 Goals",
    }

    odds_key = market  # bookmaker_odds dict uses same keys
    model_prob = prediction.get(market_prob_key.get(market, ""), 0.0)

    # Extract edge for this market (from per-key edges dict or best_edge)
    edges = prediction.get("edges", {})
    edge = edges.get(market, 0.0)

    # Bookmaker odds for this market
    bm_odds = prediction.get("bookmaker_odds") or {}
    book_odds = bm_odds.get(odds_key)

    # Fair implied probability (approx from edge + model_prob)
    fair_implied = round(model_prob - edge, 4) if edge else None

    # Fallback flags
    flags = prediction.get("fallback_flags", {})

    # Data quality score (1.0 = perfect, 0.0 = all fallbacks)
    quality_penalties = sum([
        flags.get("used_xg_fallback", False),
        flags.get("used_dc_fallback", False),
        flags.get("used_global_model", False),
        flags.get("used_approx_devig", False),
    ])
    data_quality_score = 1.0 - (quality_penalties * 0.25)

    return {
        # Match context
        "match":            f"{match_info.get('home_team', '')} vs {match_info.get('away_team', '')}",
        "home_team":        match_info.get("home_team", ""),
        "away_team":        match_info.get("away_team", ""),
        "league":           match_info.get("league", ""),
        "competition_code": match_info.get("competition_code", ""),
        "match_date":       match_info.get("match_date", ""),
        "market":           market,
        "market_label":     market_label.get(market, market),

        # Model outputs
        "model_probability":  round(model_prob, 4),
        "fair_implied":       fair_implied,
        "bookmaker_odds":     book_odds,
        "edge":               round(edge, 4),
        "confidence":         prediction.get("confidence", 0.0),
        "stars":              prediction.get("stars", 1),
        "best_edge":          prediction.get("best_edge", 0.0),
        "bet_eligible":       prediction.get("bet_eligible", False),
        "calibrated":         prediction.get("calibrated", False),
        "dc_available":       prediction.get("dc_available", False),
        "league_model_used":  prediction.get("league_model_used", False),

        # Fallback quality flags
        "fallback_flags":     flags,
        "data_quality_score": round(data_quality_score, 2),

        # Historical performance for this segment
        "historical_segment_roi":  historical_segment_roi,
        "historical_segment_bets": historical_segment_bets,

        # CLV performance
        "clv_history":   clv_history,
        "clv_beat_rate": clv_beat_rate,

        # Match context
        "injury_notes":  injury_notes or [],
        "home_injuries": prediction.get("home_injuries", []),
        "away_injuries": prediction.get("away_injuries", []),

        # xG context
        "xg_home": prediction.get("xg_home"),
        "xg_away": prediction.get("xg_away"),

        # Adjustments applied
        "adjustments": prediction.get("adjustments", []),
    }


def build_packets_all_markets(
    prediction: dict,
    match_info: dict,
    historical_roi_by_segment: Optional[dict] = None,
    clv_stats_by_market: Optional[dict] = None,
) -> list[dict]:
    """
    Build packets for every market that has an edge.
    Returns only markets where edge > 0.
    """
    markets = ["home", "draw", "away", "over25", "btts", "over35"]
    packets = []
    edges = prediction.get("edges", {})
    historical_roi_by_segment = historical_roi_by_segment or {}
    clv_stats_by_market = clv_stats_by_market or {}

    for market in markets:
        if edges.get(market, 0) > 0:
            comp = match_info.get("competition_code", "")
            seg_key = f"{comp}|{market}"
            seg_roi = historical_roi_by_segment.get(seg_key)
            seg_bets = 0

            clv_data = clv_stats_by_market.get(market, {})
            clv_avg = clv_data.get("avg_clv")
            clv_rate = clv_data.get("positive_rate")

            packets.append(build_packet(
                prediction=prediction,
                match_info=match_info,
                market=market,
                historical_segment_roi=seg_roi,
                historical_segment_bets=seg_bets,
                clv_history=clv_avg,
                clv_beat_rate=clv_rate,
                injury_notes=prediction.get("adjustments", []),
            ))

    return packets
