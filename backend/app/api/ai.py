"""
AI Decision Layer API endpoints.

Routes:
  POST /api/ai/analyze          — analyze a single prediction packet
  GET  /api/ai/best-bets        — today's top AI-graded bets
  GET  /api/ai/performance      — AI decision performance summary
  GET  /api/ai/decisions        — recent AI decisions (filterable)
  GET  /api/ai/evaluation       — full evaluation report
"""
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/ai", tags=["ai"])


class AnalyzeRequest(BaseModel):
    prediction: dict
    match_info: dict
    market: Optional[str] = None          # None = analyze all markets
    clv_stats_by_market: Optional[dict] = None
    historical_roi_by_segment: Optional[dict] = None


@router.post("/analyze")
async def analyze_prediction(body: AnalyzeRequest):
    """
    Run the AI Decision Layer on a prediction packet.

    If market is specified, returns a single market analysis.
    If market is None, returns analysis for all markets with edge.
    """
    try:
        from ai_layer.recommendation_service import analyze_market, analyze_all_markets
        from app.services.clv_tracker import get_clv_stats

        clv = body.clv_stats_by_market
        if clv is None:
            stats = get_clv_stats(days=90)
            clv = stats.get("by_market", {})

        if body.market:
            result = analyze_market(
                prediction=body.prediction,
                match_info=body.match_info,
                market=body.market,
                clv_history=clv.get(body.market, {}).get("avg_clv"),
                clv_beat_rate=clv.get(body.market, {}).get("positive_rate"),
            )
            return result
        else:
            return analyze_all_markets(
                prediction=body.prediction,
                match_info=body.match_info,
                clv_stats_by_market=clv,
                historical_roi_by_segment=body.historical_roi_by_segment or {},
            )
    except Exception as e:
        raise HTTPException(500, f"AI analysis failed: {e}")


@router.get("/best-bets")
async def get_best_bets(
    min_score: float = Query(6.0, description="Minimum AI score threshold"),
    min_grade: str = Query("C", description="Minimum grade A/B/C/D/F"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Return today's best-graded bets from the AI decisions log.

    Pulls from the today predictions cache, runs AI analysis on each,
    and returns the top bets sorted by score.
    """
    try:
        from app.api.predictions import _today_cache, _today_str
        from ai_layer.recommendation_service import analyze_all_markets
        from app.services.clv_tracker import get_clv_stats

        date_str = _today_str()
        cached = _today_cache.get(date_str, {})
        predictions = cached.get("predictions", [])

        if not predictions:
            return {
                "status": "no_predictions",
                "best_bets": [],
                "date": date_str,
                "message": "No predictions available for today yet.",
            }

        clv_stats = get_clv_stats(days=90)
        clv_by_market = clv_stats.get("by_market", {})

        grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
        min_grade_num = grade_order.get(min_grade.upper(), 3)

        best_bets = []

        for item in predictions:
            pred = item.get("prediction", {})
            if pred.get("error") or not pred.get("predicted_outcome"):
                continue

            match_info = {
                "home_team":       item.get("home_team", ""),
                "away_team":       item.get("away_team", ""),
                "league":          item.get("competition", ""),
                "competition_code": item.get("competition_code", ""),
                "match_date":      item.get("match_date", ""),
            }

            result = analyze_all_markets(
                prediction=pred,
                match_info=match_info,
                clv_stats_by_market=clv_by_market,
            )

            best = result.get("best_recommendation")
            if best and best.get("eligible"):
                score = best.get("score", 0.0)
                grade = best.get("grade", "F")
                if score >= min_score and grade_order.get(grade, 0) >= min_grade_num:
                    best_bets.append({
                        "home_team":        item.get("home_team"),
                        "away_team":        item.get("away_team"),
                        "home_team_crest":  item.get("home_team_crest"),
                        "away_team_crest":  item.get("away_team_crest"),
                        "competition":      item.get("competition"),
                        "competition_code": item.get("competition_code"),
                        "match_date":       item.get("match_date"),
                        "ai":               best,
                        "all_markets":      result.get("all_markets", []),
                        "prediction_summary": {
                            "predicted_outcome": pred.get("predicted_outcome"),
                            "confidence":        pred.get("confidence"),
                            "stars":             pred.get("stars"),
                            "best_edge":         pred.get("best_edge"),
                            "over25_prob":       pred.get("over25_prob"),
                            "btts_prob":         pred.get("btts_prob"),
                            "xg_home":           pred.get("xg_home"),
                            "xg_away":           pred.get("xg_away"),
                            "bet_eligible":      pred.get("bet_eligible"),
                            "fallback_flags":    pred.get("fallback_flags"),
                        },
                    })

        # Sort by score descending
        best_bets.sort(key=lambda x: x["ai"]["score"], reverse=True)

        return {
            "status":    "ok",
            "date":      date_str,
            "best_bets": best_bets[:limit],
            "total":     len(best_bets),
        }

    except Exception as e:
        raise HTTPException(500, f"Best bets failed: {e}")


@router.get("/performance")
async def ai_performance(days: int = Query(30)):
    """Return AI decision performance stats (grade accuracy, ROI by grade/market)."""
    try:
        from ai_layer.learning_engine import get_performance_summary
        return get_performance_summary(days=days)
    except Exception as e:
        raise HTTPException(500, f"Performance stats failed: {e}")


@router.get("/decisions")
async def recent_decisions(
    limit: int = Query(20, ge=1, le=100),
    eligible_only: bool = Query(True),
):
    """Return recent AI decisions from the log."""
    try:
        from ai_layer.learning_engine import get_recent_decisions
        return {
            "decisions": get_recent_decisions(limit=limit, eligible_only=eligible_only),
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load decisions: {e}")


@router.get("/evaluation")
async def evaluation_report(days: int = Query(365)):
    """
    Full evaluation report:
      ROI by market, league, odds bucket, edge bucket, confidence bucket,
      fallback status, calibration table, CLV summary, rolling performance.
    """
    try:
        from ml.evaluation_report import generate_report
        return generate_report(days=days)
    except Exception as e:
        raise HTTPException(500, f"Evaluation report failed: {e}")
