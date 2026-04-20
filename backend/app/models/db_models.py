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


class AIDecision(Base):
    """AI Decision Layer output — one row per market per prediction."""
    __tablename__ = "ai_decisions"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    match_id = Column(String, index=True)
    match_date = Column(String)
    league = Column(String)
    home_team = Column(String)
    away_team = Column(String)
    market = Column(String)                 # home / draw / away / over25 / btts / over35

    # AI output
    recommendation = Column(String)         # STRONG BET / BET / SMALL BET / WATCHLIST / PASS / AVOID
    grade = Column(String)                  # A / B / C / D / F
    score = Column(Float)                   # 0–10 weighted composite score
    risk_level = Column(String)             # LOW / MEDIUM / HIGH

    # Inputs snapshot
    model_prob = Column(Float)
    fair_implied = Column(Float, nullable=True)
    edge = Column(Float, nullable=True)
    bookmaker_odds = Column(Float, nullable=True)
    confidence = Column(Float)
    bet_eligible = Column(Boolean, default=False)

    # Fallback quality flags
    used_xg_fallback = Column(Boolean, default=False)
    used_dc_fallback = Column(Boolean, default=False)
    used_global_model = Column(Boolean, default=False)
    used_approx_devig = Column(Boolean, default=False)

    # Reasoning
    reasoning = Column(JSON, nullable=True)   # list of reason strings
    warnings = Column(JSON, nullable=True)    # list of warning strings
    stake_modifier = Column(Float, default=1.0)

    # Post-settlement (filled after result is known)
    actual_outcome = Column(Boolean, nullable=True)    # True = won / False = lost
    pnl = Column(Float, nullable=True)
    settled_at = Column(DateTime, nullable=True)


class SegmentStats(Base):
    """Rolling performance by segment (market × league × odds-band)."""
    __tablename__ = "segment_stats"

    id = Column(Integer, primary_key=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    segment_key = Column(String, unique=True, index=True)  # e.g. "PL|over25|1.60-1.80"
    market = Column(String)
    league = Column(String)
    odds_band = Column(String)
    confidence_band = Column(String, nullable=True)

    # Performance
    bets = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    roi = Column(Float, nullable=True)
    avg_clv = Column(Float, nullable=True)
    avg_edge = Column(Float, nullable=True)
    hit_rate = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)

    # Data quality
    pct_with_clv = Column(Float, nullable=True)
    pct_exact_devig = Column(Float, nullable=True)
