# SoccerBet AI

A soccer match outcome prediction website powered by a machine learning model (XGBoost).
Predicts **Home Win / Draw / Away Win** for upcoming fixtures across top European leagues.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite + TailwindCSS |
| Backend | FastAPI (Python) |
| ML Model | XGBoost + scikit-learn |
| Database | SQLite |
| Data Source | football-data.org (free tier) |
| Deployment | Docker Compose on Hetzner VPS |

## Getting Started (Local)

### 1. Get a free API key
Sign up at https://www.football-data.org/client/register — the free tier covers all top leagues.

### 2. Backend setup
```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your API key
```

### 3. Train the model
```bash
cd backend
python -m ml.train
```
This fetches historical results and trains the XGBoost model (~5 minutes due to API rate limits).

### 4. Start the backend
```bash
cd backend
uvicorn app.main:app --reload
```
API runs at http://localhost:8000 — docs at http://localhost:8000/docs

### 5. Frontend setup
```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```
Frontend runs at http://localhost:5173

---

## Deployment on Hetzner VPS

```bash
# On your VPS (Ubuntu)
git clone <your-repo>
cd sports-bet-analysis

# Set env vars
cp backend/.env.example backend/.env
nano backend/.env   # Add your API key

# Train model first (or copy model.joblib from local)
docker compose run --rm backend python -m ml.train

# Start everything
docker compose up -d
```

Site will be live on port 80.

### Retrain the model (cron job on VPS)
```bash
# Add to crontab to retrain weekly
0 3 * * 0 cd /path/to/project && docker compose run --rm backend python -m ml.train
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/matches/upcoming` | Upcoming fixtures |
| GET | `/api/matches/results` | Recent results |
| GET | `/api/matches/standings` | League table |
| GET | `/api/predictions/upcoming` | Upcoming fixtures + predictions |
| POST | `/api/predictions/predict` | Predict a specific match |

Full interactive docs: `http://localhost:8000/docs`

## ML Model

**Features used:**
- Team form (last 5 & 10 matches — points per game)
- Average goals scored / conceded
- Head-to-head record (last 5 meetings)
- League position & points per game
- Position difference, PPG difference, form difference

**Target:** Home Win (0) / Draw (1) / Away Win (2)

**Algorithm:** XGBoost classifier with 5-fold cross-validation
