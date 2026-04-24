# AI Decision Layer — Design & Architecture

**Author:** Craig Ataide  
**Date:** April 2026

---

## Overview

The system has three layers:

```
┌─────────────────────────────────┐
│  Layer 3: Execution             │  Bet logging, bankroll, P&L tracking
├─────────────────────────────────┤
│  Layer 2: AI Analyst            │  Interpretation, judgment, grades, reasoning
├─────────────────────────────────┤
│  Layer 1: Quant Engine          │  Math, statistics, probability generation
└─────────────────────────────────┘
```

> "Your first layer predicts. Your second layer thinks. Your third layer executes. That is how this becomes powerful."

The AI layer does not replace the betting model. It sits above it and acts as an analyst + risk manager. It reads structured prediction packets from Layer 1 and outputs structured bet recommendations with grades, confidence scores, and reasoning.

---

## Layer 1 Output → Layer 2 Input: The Prediction Packet

The AI layer receives a structured JSON packet, not unstructured text. Giving the AI "Arsenal vs Chelsea who wins?" produces guessing. Giving it edge, fallbacks, historical bucket ROI, and CLV produces useful analysis.

**Prediction packet structure (from `backend/ml/predict.py`):**

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "competition_code": "PL",
  "match_date": "2026-04-24",

  "home_win_prob": 0.4812,
  "draw_prob": 0.2831,
  "away_win_prob": 0.2357,
  "predicted_outcome": "HOME",
  "confidence": 0.4812,
  "stars": 3,

  "over25_prob": 0.6203,
  "btts_prob": 0.5711,
  "best_edge": 0.0623,
  "value_bets": ["Home Win (54% vs 48% fair)"],
  "kelly_stakes": {"home_win": 0.072},

  "bet_eligible": true,
  "fallback_flags": {
    "used_xg_fallback": false,
    "used_dc_fallback": false,
    "used_global_model": false,
    "used_approx_devig": false
  },
  "calibrated": true,
  "league_model_used": true
}
```

---

## Layer 2 Output: AI Recommendation

```json
{
  "recommendation": "BET",
  "grade": "B",
  "score": 7.4,
  "risk_level": "MEDIUM",
  "market": "home_win",
  "stake_modifier": 1.0,
  "reasoning": [
    "Edge of 6.2% exceeds the 5% minimum threshold",
    "League-specific model active — no global fallback",
    "Dixon-Coles and XGBoost agree on home win direction",
    "Calibrated probabilities — Brier score within acceptable range"
  ],
  "warnings": [
    "Confidence at 0.48 — marginal eligibility"
  ],
  "eligible": true
}
```

---

## Recommendation Types

| Label | Meaning |
|-------|---------|
| `STRONG BET` | Score ≥ 8.5 — high edge, high confidence, clean data |
| `BET` | Score ≥ 7.0 — clear edge, good data quality |
| `SMALL BET` | Score ≥ 6.0 — positive but weaker signal |
| `WATCHLIST` | Score ≥ 5.0 — monitor for line movement |
| `PASS` | Below threshold — no action |
| `AVOID` | Negative signal or data quality failure |

---

## AI Scoring Formula

```
score = (edge_score      × 0.35)
      + (confidence_score × 0.25)
      + (historical_score × 0.20)
      + (clv_score        × 0.10)
      + (data_quality     × 0.10)
```

Each component is normalised to 0–10:
- **edge_score**: maps `best_edge` from 0.0–0.15+ onto 0–10
- **confidence_score**: maps `confidence` from 0.50–0.80+ onto 0–10
- **historical_score**: derived from backtest ROI for this league/market/odds bucket
- **clv_score**: derived from CLV log for this type of prediction
- **data_quality**: 10 if no fallbacks used; lower if any flag is set

**Implemented in:** `backend/app/api/ai.py`

---

## AI Constraints — What the AI Must Never Do

1. **Never invent stats** — all inputs must come from the Layer 1 packet
2. **Never ignore model outputs** — reasoning must be grounded in the data
3. **Never guess injuries** — use only what is in the injury data fields
4. **Never override hard filters** — if `bet_eligible = false`, no recommendation is issued
5. **Never hallucinate historical performance** — CLV and ROI must come from the ledger

---

## File Structure

```
backend/app/api/ai.py           — endpoints: /best-bets, /analyze, /performance, /decisions
backend/app/services/
    prediction_service.py       — calls AI layer after ML prediction
    evaluator.py                — logs decisions to ledger
    clv_tracker.py              — CLV calculation and logging
```

---

## Database Tables

| Table | Purpose |
|-------|---------|
| `predictions` | Every match prediction with full packet |
| `ai_decisions` | Every AI recommendation with grade + reasoning |
| `bet_results` | Settled bets: stake, odds, closing odds, result, P&L |
| `segment_stats` | ROI by league / market / season / odds bucket |
| `clv_logs` | CLV at settlement for every tracked prediction |

---

## Build Phases

| Week | Deliverable |
|------|-------------|
| 1 | Prediction packet JSON fully wired and logged |
| 2 | Rules engine (eligibility gate + filters) |
| 3 | Recommendation scoring formula |
| 4 | Dashboard grades (A–F, STRONG BET labels) |
| 5 | LLM explanation system |
| 6 | Results tracking and learning layer |

---

## Local vs API LLM Strategy

- **High-volume analysis (all matches):** Local LLM (Llama 3 8B / Qwen 7B / Mistral 7B) — fast, free, runs on-device
- **Premium analysis (Best Bets page):** API model (Claude Sonnet) — higher reasoning quality, costs per call
- **Hybrid rule:** Rules engine decides eligibility; LLM explains and prioritises eligible bets only
