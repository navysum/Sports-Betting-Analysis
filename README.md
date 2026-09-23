# Sports Betting Analysis

A football prediction platform built for the serious bettor. It combines a Dixon-Coles statistical model with an XGBoost ensemble, runs Monte Carlo match simulations, tracks Closing Line Value (CLV) against Pinnacle, and surfaces value bets automatically — all refreshed daily and served through a React dashboard.

Built with Neil, from April to June 2026.

**Status (23 Sep 2026): the frontend is live, but the model data is stale.** The
dashboard is up at https://sports-betting-analysis-two.vercel.app, with the
backend on Render. GitHub **disabled both scheduled workflows for inactivity**:
*Daily Data Cache Refresh + CLV Snapshot* last ran on 6 Aug 2026, and *Weekly
Model Retrain* on 9 Aug 2026. Until they are re-enabled under **Actions**, cached
data and model parameters are not refreshed. The last code change was on 6 Jun 2026.

---

## What it does

| Feature | Detail |
| --- | --- |
| **Match predictions** | Home/Draw/Away, Over 2.5 goals, BTTS, Over 3.5 goals — with confidence ratings and value badges |
| **Value bet detection** | Compares model probability to devigged bookmaker fair odds; flags bets with positive expected value |
| **Monte Carlo simulator** | 50,000-simulation DC-corrected score grid per match; Asian Handicap table; EV badges; parameter uncertainty sampling |
| **CLV tracking** | Logs model probability vs Pinnacle opening and closing implied odds; measures real edge over time |
| **Distributions page** | Visualises goal distributions and score probabilities from the underlying model |
| **Results & accuracy** | Tracks historical prediction accuracy by market (7-day, 30-day, all-time) |
| **Standings** | Live league tables fetched from football-data.org |
| **Team pages** | Per-team form, xG, H2H history |
| **Daily auto-refresh** | APScheduler cron at 06:00 UTC preloads predictions; GitHub Actions keep Dixon-Coles params fresh |

---

## Architecture

```
Browser (React + Vite + Tailwind)
        |
        |    REST / JSON
        v
FastAPI backend (Python 3.11)
    |-- APScheduler (daily 06:00 UTC prediction preload)
    |-- SQLite via SQLAlchemy (async)
    |-- Prediction pipeline
    |       |-- XGBoost ensemble   (result / goals / BTTS / over35 models)
    |       |-- Dixon-Coles model  (tau-corrected Negative Binomial + Poisson score grid)
    |       |-- Blend optimiser    (Optuna — weights per market, auto-runs post-retrain)
    |-- Odds API client            (The Odds API — value bet detection + CLV)
    |-- Injury service             (API-Football via RapidAPI)
    |-- CLV tracker                (model prob vs Pinnacle closing line)
        |
        |-- football-data.org API  (fixtures, results, standings, H2H)
        |-- The Odds API           (bookmaker odds + Pinnacle sharp reference)
        |-- API-Football           (injury reports)
```

---

## Repository layout

```
Sports-Betting-Analysis/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app + APScheduler
│   │   ├── api/               # matches, predictions, admin, ai routes
│   │   ├── services/          # football-data, odds, injuries (API-Football), CLV tracker,
│   │   │                      # prediction service, evaluator, API cache, scrapers
│   │   ├── models/            # SQLAlchemy models
│   │   └── config.py, database.py, utils/
│   ├── ml/                    # Dixon-Coles, Elo, features, XGBoost models + calibrators,
│   │                          # blend optimiser, backtest, evaluation report, trainer
│   ├── ai_layer/              # AI decision layer: packet builder, rules, scoring, recommendations
│   ├── data/                  # fitted parameters: Dixon-Coles, Elo, blend weights, team aliases
│   └── Dockerfile
├── frontend/                  # React + Vite + Tailwind: pages (best bets, CLV, distributions, …), components
├── .github/workflows/         # daily-refresh.yml, weekly-retrain.yml (both disabled by GitHub for inactivity)
├── docs/                      # AI decision layer design, model review plan, roadmap, to-do checklist
├── hosting/HOSTING.md         # Render (backend) + Vercel (frontend) deployment guide
├── render.yaml, docker-compose.yml
```

---

## Model stack

### Dixon-Coles

- Score grid fitted using a **Negative Binomial (NB2) PMF** alongside the standard Poisson, capturing over-dispersion in goal-scoring
- **Per-league rho** (`rho_by_league`): low-scoring leagues (e.g. Ligue 1) use a stronger τ-correction for {0-0, 1-0, 0-1, 1-1} scorelines than high-scoring leagues (e.g. Bundesliga)
- **Convergence retry**: if the first SciPy `minimize` run fails to converge, parameters are perturbed and the optimiser retries once
- BTTS derived purely from the score grid (not a post-hoc xG correction)
- Home advantage and per-league rho updated nightly via GitHub Actions

### XGBoost ensemble

- **Result model** — 3-class (HOME / DRAW / AWAY), adaptive calibration (sigmoid < 2000 cal samples, isotonic ≥ 2000)
- **Goals model** — binary (Over 2.5), same adaptive calibration
- **BTTS model** — binary, adaptive calibration
- **Over 3.5 model** — binary, adaptive calibration
- **Training**: `TimeSeriesSplit(n_splits=5)` to prevent temporal leakage; early stopping; balanced sample weights
- **Per-league calibrators**: separate calibrators per competition (PL, BL1, SA, FL1, PD, …)
- **Feature set**: rolling form (5-match, λ=0.80 decay), xG for/against (Understat, scaled 0.80×), ELO ratings (K=20, seasonal mean reversion), H2H record, home advantage flag
- **Blend weights**: per-market DC vs XGBoost blending, optimised on a temporal holdout (0.01-step grid, not 0.05) and stored in `backend/data/blend_weights.json`

### ELO ratings

- K-factor 20 (down from 32 — reduces volatility on large-margin wins)
- **Seasonal mean reversion**: at the start of each season, team ratings are pulled 20% toward the league mean to partially absorb squad changes
- Ratings saved to `backend/data/elo_ratings.json` after every training run and reloaded hourly at inference (TTL-based refresh)

### Monte Carlo simulator

- 50,000 samples per match from the DC τ-corrected CDF score grid
- **Parameter uncertainty**: λ and μ re-drawn from LogNormal(σ=0.12) every 2,000 sims
- Outputs: 1X2 probs, Asian Handicap table (±0.25 / ±0.50 / ±0.75 / ±1.00 / ±1.25 / ±1.50), Over/Under lines, BTTS, EV badges at given odds
- Convergence tracked live as simulations run

### CLV tracker

- Records model probability and Pinnacle opening odds at prediction time using the stable API match ID (not a name-derived abbreviation)
- After market close, `update_closing()` records Pinnacle closing implied probability
- `CLV = model_prob − pinnacle_closing_implied`
- Markets tracked: 1X2 (home/draw/away) + Over 2.5 goals
- Persistent append-only log: `backend/data/clv_log.json`

### Backtesting

- Temporal holdout: most recent 30% of FDCO historical data (no random shuffling)
- Full production pipeline: blended DC + XGBoost probabilities across all four markets
- Devigged fair implied probabilities (Shin method for 1X2, exact two-outcome for binary markets)
- Three staking strategies: flat, value (≥3% edge over fair), fractional Kelly (¼ Kelly, 5% bankroll cap)
- All win-rate statistics include **95% Wilson score confidence intervals**

---

## Competitions covered

| Code | League |
| ------ | -------- |
| PL | English Premier League |
| PD | Spanish La Liga |
| BL1 | German Bundesliga |
| SA | Italian Serie A |
| FL1 | French Ligue 1 |
| ELC | Championship |
| DED | Eredivisie |
| PPL | Primeira Liga |
| CL | UEFA Champions League |

---

## Prerequisites

| Service | Free tier | Used for |
| --- | --- | --- |
| [football-data.org](https://www.football-data.org/client/register) | Free | Fixtures, results, standings, H2H |
| [The Odds API](https://the-odds-api.com) | 500 req/month | Bookmaker odds, Pinnacle CLV reference |
| [API-Football (RapidAPI)](https://rapidapi.com/api-sports/api/api-football) | 100 req/day | Injury reports |
| Telegram bot (optional) | Free | Alerts via @BotFather |

---

## Local setup

```bash
# 1. Clone
git clone https://github.com/navysum/Sports-Betting-Analysis.git
cd Sports-Betting-Analysis

# 2. Backend
cd backend
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy .env and fill in your API keys
cp .env.example .env

# Start the API
uvicorn app.main:app --reload
# → http://localhost:8000

# 3. Frontend (separate terminal)
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

The backend serves predictions immediately on startup — models are pre-trained (`.joblib` files committed to the repo). No training required to run locally.

---

## Training models

Run this inside `backend/` with your virtual environment active:

```bash
python -m ml.train
```

Training takes 20–30 minutes and fetches data from football-data.org (requires `FOOTBALL_DATA_API_KEY`). When complete, it:
1. Writes updated `.joblib` model files to `backend/ml/`
2. Writes per-league calibrators (`result_calibrator_PL.joblib`, etc.)
3. Auto-runs the blend weight optimiser and updates `backend/data/blend_weights.json`

Commit and push the updated models to redeploy with fresh weights:

```bash
git add backend/ml/*.joblib backend/data/blend_weights.json \
        backend/data/dixon_coles_params.json backend/data/elo_ratings.json
git commit -m "Retrain models - $(date +%Y-%m-%d)"
git push origin main
```

---

## Environment variables

### Backend (`backend/.env`)

| Variable | Required | Description |
| --- | --- | --- |
| `FOOTBALL_DATA_API_KEY` | ✓ | football-data.org free key |
| `ODDS_API_KEY` | Optional | The Odds API key (value bets + CLV) |
| `API_FOOTBALL_KEY` | Optional | RapidAPI key for injury reports |
| `TELEGRAM_TOKEN` | Optional | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | Optional | Target chat/group ID |
| `TELEGRAM_USER_ID` | Optional | Your Telegram user ID (admin commands) |
| `DATABASE_URL` | Optional | Defaults to `sqlite+aiosqlite:///./data/soccerbet.db` |
| `TIMEZONE` | Optional | Defaults to `Europe/London` |

### Frontend (`frontend/.env`)

| Variable | Required | Description |
| --- | --- | --- |
| `VITE_API_URL` | ✓ | Backend base URL, e.g. `http://localhost:8000/api` |

---

## Docker (local or server)

```bash
# Build and run both services
docker-compose up --build

# Frontend → http://localhost:80
# Backend  → http://localhost:8000 (internal only, proxied via nginx)
```

Set `FOOTBALL_DATA_API_KEY` and other secrets in `backend/.env` before running.

---

## Deployment

See [`hosting/HOSTING.md`](hosting/HOSTING.md) for step-by-step instructions to deploy:
- **Backend** → Render.com (free web service, Docker runtime)
- **Frontend** → Vercel (free, auto-deploys on push to `main`)

---

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/predictions/today` | Today's predictions for all competitions |
| `GET` | `/api/predictions/{match_id}` | Single match prediction detail |
| `GET` | `/api/matches` | Upcoming fixtures |
| `GET` | `/api/matches/{match_id}` | Single match with H2H |
| `GET` | `/api/standings/{competition}` | League table |
| `GET` | `/accuracy` | Prediction accuracy stats (7d / 30d / all) |
| `GET` | `/health` | Health check |
| `POST` | `/api/admin/retrain` | Trigger model retrain (background) |
| `GET` | `/api/admin/clv` | CLV performance summary |
| `GET` | `/api/admin/odds-quota` | Odds API requests remaining |

---

## Project structure

```
Sports-Betting-Analysis/
├── backend/
│   ├── app/
│   │   ├── api/                     # FastAPI routers (matches, predictions, admin)
│   │   ├── models/                  # SQLAlchemy DB models
│   │   ├── services/                # Business logic (prediction_service, odds_api,
│   │   │                            #   clv_tracker, injury_service, scraper, …)
│   │   ├── config.py                # Pydantic settings
│   │   ├── database.py              # Async SQLite init
│   │   └── main.py                  # FastAPI app + APScheduler setup
│   ├── ml/
│   │   ├── train.py                 # Full training pipeline (TimeSeriesSplit, per-league cal)
│   │   ├── predict.py               # Inference (per-league calibrators, devig, DC blend)
│   │   ├── dixon_coles.py           # DC model (NB2 + Poisson, per-league rho, tau-correction)
│   │   ├── features.py              # Feature engineering (form decay, H2H priors, xG)
│   │   ├── fdco_trainer.py          # Football-Data.co.uk training data loader
│   │   ├── optimize_blend.py        # Blend weight optimiser (0.01-step grid)
│   │   ├── backtest.py              # Historical backtesting (full pipeline, Wilson CIs)
│   │   ├── elo.py                   # ELO rating system (K=20, seasonal reversion)
│   │   └── *.joblib                 # Trained models (committed, rebuilt on retrain)
│   ├── data/
│   │   ├── blend_weights.json       # DC vs XGBoost weights per market
│   │   ├── dixon_coles_params.json  # Fitted DC parameters (NB2 + per-league rho)
│   │   ├── elo_ratings.json         # Current ELO ratings
│   │   └── team_aliases.json        # Team name normalisation map
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── pages/                   # HomePage, PredictionsPage, MonteCarloPage,
│   │   │                            #   ResultsPage, StatsPage, StandingsPage,
│   │   │                            #   DistributionsPage, MatchesPage, TeamPage
│   │   ├── components/              # PredictionCard, CompetitionSelector, FormDots, …
│   │   ├── services/api.js          # All API calls
│   │   └── utils/time.js            # Kick-off time formatting
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── hosting/
│   └── HOSTING.md                   # Render + Vercel deployment guide
├── .github/
│   └── workflows/
│       └── daily-refresh.yml        # Nightly Dixon-Coles parameter refresh
├── docker-compose.yml
├── render.yaml                      # Render.com blueprint
└── .gitignore
```
