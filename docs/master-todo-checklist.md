# Sports Betting Analysis — Master Build Checklist

**Author:** Craig Ataide  
**Date:** April 2026

This is the full aspirational roadmap across 10 phases. Items are unchecked unless noted.

---

## Golden Rules

- [ ] Never force bets
- [ ] Trust data over emotion
- [ ] Trust ROI over win rate
- [ ] Trust CLV over hype
- [ ] If unclear → PASS
- [ ] If edge weak → PASS
- [ ] If data poor → PASS
- [ ] Improve weekly

---

## Immediate Priority (Do First)

1. [ ] Full backtest report (ROI by market / league / season / odds bucket / edge bucket / confidence bucket / fallback status)
2. [ ] Better data layer (team name mapping cleanup, injury data reliability, xG source consistency)
3. [ ] Prediction packet wired end-to-end with all fields
4. [ ] AI rules engine: eligibility gate tightened, segment filters added
5. [ ] Dashboard improvements: calibration plots, CLV chart, segment ROI table

---

## Phase 1 — Foundation & Infrastructure

- [ ] Folder structure consistent across backend / ml / data / docs
- [ ] Environment management (`.env`, `settings.py`, secrets never committed)
- [ ] Structured logging (log level, timestamp, component prefix)
- [ ] Error handling: all external calls wrapped, fallbacks explicit
- [ ] SQLite or DuckDB for predictions / decisions / bet results storage
- [ ] Automated daily backups of model weights and database
- [ ] Unit tests for feature engineering and devigging logic

---

## Phase 2 — Data Layer

- [x] Football-Data.co.uk CSVs downloaded and parsed (FDCO historical data)
- [x] football-data.org live fixtures and standings (free tier)
- [x] API-Football injuries (100 req/day free tier)
- [x] Team name mapping system (`utils/team_names.py`)
- [x] Disk-based API response cache with TTL (`services/api_cache.py`)
- [x] Daily 6am UTC preload job (APScheduler)
- [ ] Odds history storage (closing odds logged for CLV calculation)
- [ ] Weekly data refresh for team histories and standings
- [ ] Monitor API quota usage and alert before hitting limits

---

## Phase 3 — Feature Engineering

### Core features (implemented)
- [x] Home/away form (last 5/10 matches)
- [x] Goals scored / conceded
- [x] League position and points per game
- [x] ELO ratings with K=20 and seasonal mean reversion
- [x] Head-to-head record
- [x] Home advantage weighting
- [x] xG (Understat/Sofascore)

### Advanced features (to build)
- [ ] Shot accuracy and conversion rate
- [ ] Set-piece danger (corners, free kicks in danger zones)
- [ ] Possession percentage and pressing intensity
- [ ] Player absence weighting (see `docs/advanced-features-roadmap.md`)
- [ ] Manager effects (new manager bounce, sack risk)
- [ ] Motivation context (title race, relegation, dead rubber)
- [ ] Derby / rivalry flag

---

## Phase 4 — Prediction Models

- [x] XGBoost per market: 1X2, Over 2.5, BTTS, Over 3.5
- [x] Dixon-Coles Poisson model
- [x] XGBoost + Dixon-Coles blended ensemble
- [x] League-specific models (separate weights per competition)
- [x] Isotonic regression calibration
- [x] TimeSeriesSplit cross-validation (no temporal leakage)
- [x] Fallback flags when global model used
- [ ] CatBoost and LightGBM added to ensemble
- [ ] Market-implied probability model (devigged closing odds as baseline)
- [ ] Drift detection (ROI / calibration / CLV degradation alerts)
- [ ] Season-by-season model comparison

---

## Phase 5 — Value Detection

- [x] Devigging engine (Shin method for 1X2; binary devig for O/U and BTTS)
- [x] Multi-market edge calculation
- [x] Quarter-Kelly staking with 5% bankroll cap
- [x] Bet eligibility gate (data quality + edge + confidence)
- [ ] Closing odds comparison (open vs close)
- [ ] Odds-band segmentation (1.20–1.50 / 1.50–2.00 / 2.00–3.00 / 3.00+)
- [ ] No-bet detector (explicit AVOID signal, not just absence of BET)
- [ ] Market movement alerts (steam moves, reverse line movement)

---

## Phase 6 — Backtesting

- [x] Basic backtest endpoint (`/api/admin/backtest`)
- [x] Brier score calibration metric
- [x] Flat / value / Kelly staking comparison
- [ ] Full ROI breakdown by:
  - Market (1X2 / O2.5 / BTTS / O3.5)
  - League (PL / PD / BL1 / SA / FL1 / ELC / DED / PPL)
  - Season (22/23, 23/24, 24/25)
  - Odds bucket (1.20–1.50, 1.50–2.00, 2.00–3.00)
  - Edge bucket (0.03–0.05, 0.05–0.08, 0.08–0.12, 0.12+)
  - Confidence bucket (0.50–0.60, 0.60–0.70, 0.70–0.80, 0.80+)
  - Fallback status (clean vs any fallback)
- [ ] Rolling CLV distribution chart
- [ ] Closing line simulation (realistic odds at time of bet vs close)
- [ ] Realistic stake cap simulation (£500 limit, typical bookmaker restrictions)

---

## Phase 7 — AI Decision Layer

- [x] Prediction packet JSON (all fields wired from `ml/predict.py`)
- [x] Rules engine: bet eligibility gate (`bet_eligible` flag)
- [x] AI scoring formula (score = weighted sum of edge/confidence/historical/CLV/data_quality)
- [x] Grade system: A–F grades, STRONG BET / BET / SMALL BET / WATCHLIST / PASS / AVOID
- [x] Best Bets page with AI grades displayed
- [ ] LLM explanation system (why this bet / why not this bet)
- [ ] Daily best bets report (top 5, safest 3, value picks)
- [ ] Results tracking: log settled bets with closing odds and P&L
- [ ] Learning layer: use settled results to improve scoring weights

---

## Phase 8 — Dashboard

- [x] Fixtures page (today's matches with kickoff times)
- [x] Predictions page (ML predictions with confidence bars)
- [x] Best Bets page (AI-graded eligible bets)
- [x] Results page
- [x] Standings page
- [x] Stats / backtest page
- [x] Charts / distributions page
- [x] Analytics page (CLV, model performance)
- [ ] Calibration plot (reliability diagram)
- [ ] CLV rolling chart (rolling 50-bet average)
- [ ] Segment ROI table (league × market × odds bucket)
- [ ] AI grade explanations displayed on card
- [ ] No-bet warning cards

---

## Phase 9 — Automation

- [x] Morning preload job (6am UTC, all competitions)
- [x] Server keep-alive pings (GitHub Actions, 14-minute interval)
- [ ] Pre-match odds refresh (2 hours before kickoff)
- [ ] In-play score updates and result settlement
- [ ] Strong bet alert (Telegram/Discord when score ≥ 8.5)
- [ ] Odds movement alert (line moves > 10% in 30 minutes)
- [ ] Night job: settle yesterday's predictions against results

---

## Phase 10 — Long-Term Growth

- [ ] League-specific AI agents (separate tuning per competition)
- [ ] Bankroll manager AI (track actual P&L, adjust Kelly fraction based on recent performance)
- [ ] Multi-sport expansion (tennis, basketball)
- [ ] Subscription model (paid API access to predictions and grades)
- [ ] API white-label (provide model as an API to other developers)
- [ ] Trading history export (CSV download for tax / analysis)
