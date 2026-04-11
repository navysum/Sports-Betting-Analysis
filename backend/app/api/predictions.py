from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from app.services.prediction_service import predict_match, predict_upcoming_batch
from app.services.football_api import get_upcoming_matches, SUPPORTED_COMPETITIONS, FDORG_COMPETITIONS

router = APIRouter(prefix="/predictions", tags=["predictions"])


class PredictRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    competition_code: str = "PL"
    home_team_name: str = ""
    away_team_name: str = ""
    match_date: str = ""
    save_to_ledger: bool = False
    bookmaker_odds: Optional[dict] = None


@router.post("/predict")
async def predict_single(body: PredictRequest):
    if body.competition_code not in FDORG_COMPETITIONS:
        raise HTTPException(400, f"Unsupported competition. Choose from: {list(FDORG_COMPETITIONS)}")
    try:
        result = await predict_match(
            home_team_id=body.home_team_id,
            away_team_id=body.away_team_id,
            competition_code=body.competition_code,
            home_team_name=body.home_team_name,
            away_team_name=body.away_team_name,
            match_date=body.match_date,
            save_to_ledger=body.save_to_ledger,
            bookmaker_odds=body.bookmaker_odds,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@router.get("/upcoming")
async def predict_upcoming(
    competition: str = Query("PL"),
    days_ahead: int = Query(1, ge=1, le=14),
    save_to_ledger: bool = Query(False),
):
    """Returns upcoming matches with full multi-market predictions attached."""
    if competition not in FDORG_COMPETITIONS:
        raise HTTPException(400, "Unsupported competition.")
    try:
        results = await predict_upcoming_batch(competition, days_ahead, save_to_ledger)
        return {"competition": competition, "predictions": results}
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch predictions: {e}")
