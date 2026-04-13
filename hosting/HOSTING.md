# Sports Bet Analysis — Free Hosting Guide

**Stack:** React frontend on Vercel + FastAPI backend on Render.com  
**Cost:** $0 forever (within free tier limits)

---

## Architecture

```
User browser
    │
    ▼
Vercel (frontend)          ← static React build, global CDN, never sleeps
    │  VITE_API_URL
    ▼
Render.com (backend)       ← FastAPI + XGBoost + Dixon-Coles, free web service
    │
    ▼
Football-Data.org API      ← live match data (API key in Render env vars)
```

**Free tier limits:**
| Service | RAM | Sleep | Auto-deploy |
|---------|-----|-------|-------------|
| Render (backend) | 512 MB | After 15 min idle → ~30s cold start | On git push |
| Vercel (frontend) | N/A (CDN) | Never | On git push |

---

## Backend — Render.com

### First-time setup

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect GitHub repo: `navysum/Sports-Betting-Analysis`
3. Fill in:
   | Field | Value |
   |-------|-------|
   | Language | Docker |
   | Root Directory | *(leave blank)* |
   | Dockerfile Path | `./backend/Dockerfile` |
   | Branch | `main` |
   | Region | Frankfurt (EU Central) |

4. Under **Environment** tab, add:
   ```
   FOOTBALL_DATA_API_KEY = <your key from football-data.org>
   DATABASE_URL = sqlite+aiosqlite:///./data/soccerbet.db
   ```

5. Click **Deploy**. First build takes ~5 minutes (installs Python deps).

6. Your backend URL will be something like:  
   `https://sportsbet-api.onrender.com`

### Troubleshooting

**"No result model found" warning on startup**  
The Docker image was built before the trained `.joblib` files were committed.  
Fix: Go to Render dashboard → **Manual Deploy → Deploy latest commit**

**405 on port detection (normal)**  
```
HEAD / HTTP/1.1 405 Method Not Allowed
==> Detected new open port HTTP:8000. Restarting...
```
This is Render's port discovery — not an error. The restart takes ~10s and is expected on first deploy.

**Health check**  
Render pings `GET /health` → should return `{"status": "healthy"}`

---

## Frontend — Vercel

### First-time setup

1. Go to [vercel.com](https://vercel.com) → **New Project**
2. Import GitHub repo: `navysum/Sports-Betting-Analysis`
3. Configure:
   | Field | Value |
   |-------|-------|
   | Root Directory | `frontend` |
   | Framework Preset | Vite (auto-detected) |
   | Build Command | `npm run build` |
   | Output Directory | `dist` |

4. Under **Environment Variables**, add:
   ```
   VITE_API_URL = https://<your-render-service>.onrender.com/api
   ```
   Replace with your actual Render URL from step above.

5. Click **Deploy**. Builds in ~1 minute.

6. Your app is live at `https://<your-project>.vercel.app`

### React Router (SPA routing)

`frontend/vercel.json` already handles this — all routes redirect to `index.html` so direct links to `/predictions`, `/standings`, etc. work correctly.

---

## Redeploying

Both services auto-deploy whenever you push to `main`.

To trigger a manual redeploy (e.g. after a hotfix):
- **Render**: Dashboard → your service → **Manual Deploy → Deploy latest commit**
- **Vercel**: Dashboard → your project → **Deployments → Redeploy**

---

## Retraining Models

Models are baked into the Docker image at build time (committed to git as `.joblib` files).

### Retrain locally and push

```bash
# From backend/ directory
python -m ml.train

# After training completes (~20-30 min), commit the new models
cd ..   # repo root
git add backend/ml/*.joblib backend/data/dixon_coles_params.json backend/data/elo_ratings.json
git commit -m "Retrain models - <date>"
git push origin main
```

Render auto-deploys with the new models baked in.

### Retrain via the app (live)

1. Open the app → **Stats** page
2. Click **Retrain Model** button
3. The retrain runs in the background on the server (takes 20-30 min)
4. Progress is shown in the live log on the Stats page
5. Note: on Render free tier, models saved this way are lost on the next deploy/restart
   — always commit retrained models to git for persistence

---

## Local Development

```bash
# Backend
cd backend
source venv/bin/activate   # or venv\Scripts\activate on Windows
uvicorn app.main:app --reload
# → http://localhost:8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

Frontend dev server proxies nothing — `api.js` calls `http://localhost:8000/api` directly.  
For production, `VITE_API_URL` is set in Vercel and baked into the build.

---

## Environment Variables Reference

| Variable | Where | Value |
|----------|-------|-------|
| `FOOTBALL_DATA_API_KEY` | Render | Your key from football-data.org |
| `DATABASE_URL` | Render | `sqlite+aiosqlite:///./data/soccerbet.db` |
| `VITE_API_URL` | Vercel | `https://<render-service>.onrender.com/api` |

`FOOTBALL_DATA_API_KEY` is the only secret — never commit it to git.  
Get a free key at [football-data.org](https://www.football-data.org/client/register).
