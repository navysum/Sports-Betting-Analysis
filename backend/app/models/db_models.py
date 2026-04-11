from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, unique=True, index=True)
    name = Column(String, nullable=False)
    short_name = Column(String)
    crest_url = Column(String)


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, unique=True, index=True)
    competition = Column(String, nullable=False)
    competition_code = Column(String)
    match_date = Column(DateTime, nullable=False)
    status = Column(String, default="SCHEDULED")  # SCHEDULED, FINISHED, IN_PLAY

    home_team_id = Column(Integer, ForeignKey("teams.api_id"))
    away_team_id = Column(Integer, ForeignKey("teams.api_id"))
    home_team_name = Column(String)
    away_team_name = Column(String)

    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    result = Column(String, nullable=True)  # HOME, DRAW, AWAY

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


class Prediction(Base):
    """Legacy simple prediction — kept for API compatibility."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.api_id"), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    predicted_outcome = Column(String)  # HOME, DRAW, AWAY
    confidence = Column(Float)

    was_correct = Column(Boolean, nullable=True)

    match = relationship("Match", foreign_keys=[match_id])


class PredictionLedger(Base):
    """Full prediction record — the bot's memory and training ground."""
    __tablename__ = "prediction_ledger"

    id = Column(Integer, primary_key=True)
    match_id = Column(String, index=True)       # e.g. "EPL-2026-04-10-ARS-CHE"
    api_match_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    match_date = Column(String)
    league = Column(String)
    home_team = Column(String)
    away_team = Column(String)

    # Predictions
    predicted_result = Column(String)           # HOME / DRAW / AWAY
    result_confidence = Column(Float)
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    over25_prob = Column(Float, nullable=True)
    btts_prob = Column(Float, nullable=True)
    stars = Column(Integer, nullable=True)      # 1–5 confidence stars

    # Actuals (filled in post-match)
    actual_result = Column(String, nullable=True)
    actual_score = Column(String, nullable=True)
    actual_over25 = Column(Boolean, nullable=True)
    actual_btts = Column(Boolean, nullable=True)

    result_correct = Column(Boolean, nullable=True)
    over25_correct = Column(Boolean, nullable=True)
    btts_correct = Column(Boolean, nullable=True)

    # Feature snapshot and post-mortem
    factors_used = Column(JSON, nullable=True)          # list of factor names
    key_factors = Column(Text, nullable=True)           # human-readable summary
    post_mortem = Column(Text, nullable=True)           # filled after evaluation


class BetLog(Base):
    """Every bet logged via /bet command."""
    __tablename__ = "bet_log"

    id = Column(Integer, primary_key=True)
    logged_at = Column(DateTime, default=datetime.utcnow)

    description = Column(String, nullable=False)   # e.g. "Arsenal ML"
    odds = Column(Float, nullable=False)
    stake = Column(Float, nullable=False)
    status = Column(String, default="PENDING")     # PENDING / WON / LOST / VOID

    pnl = Column(Float, nullable=True)             # profit (+) or loss (-) when settled
    settled_at = Column(DateTime, nullable=True)
    notes = Column(String, nullable=True)


class MatchXG(Base):
    """xG data from Understat (enriches feature vectors)."""
    __tablename__ = "match_xg"

    id = Column(Integer, primary_key=True)
    api_match_id = Column(Integer, index=True, nullable=True)
    home_team = Column(String)
    away_team = Column(String)
    league = Column(String)
    season = Column(String)
    match_date = Column(String)

    home_xg = Column(Float, nullable=True)
    away_xg = Column(Float, nullable=True)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
