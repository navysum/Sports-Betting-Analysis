# Advanced Features & Long-Term Edge Roadmap

**Author:** Craig Ataide  
**Date:** April 2026

---

## Golden Rule

> "Future edge usually comes from: Better data. Better timing. Better filtering. Better discipline. Better segmentation. Not just a more complex model."

---

## Priority Order

| Priority | Feature | Category |
|----------|---------|----------|
| 1 | Predicted lineups engine | Market Intelligence |
| 2 | Player impact model | Team/Player Intelligence |
| 3 | CLV intelligence engine | Market Intelligence |
| 4 | No-bet detector | AI Decision Layer |
| 5 | Market movement tracker | Market Intelligence |
| 6 | Team style matchup engine | Team/Player Intelligence |
| 7 | Motivation/context engine | Match Context |
| 8 | Drift detection | Quant/Model Science |
| 9 | Segment confidence scoring | AI Decision Layer |
| 10 | Referee/weather layers | Environmental |

---

## Priority 1 — Market Intelligence

### Predicted Lineups Engine
- Source: API-Football projected lineups (available 1–2 hours pre-kick)
- Use case: detect rotation/rest → adjust strength estimates downward
- Implementation: pull lineup API data in pre-match job, integrate into feature vector
- Impact: **highest** — a star striker absent = 8–15% swing in goal probability

### CLV Intelligence Engine
- Track rolling CLV distribution by: league / market / odds bucket / model configuration
- Alert when segment CLV drops below 0 over rolling 50-bet window
- Suspend betting in that segment until CLV recovers
- Implementation: extend `clv_tracker.py` with segment aggregation and alerting

### Market Movement Tracker
- Monitor odds movement from opening line to closing line
- Flag "steam moves" (sharp money signals: rapid movement against public)
- Flag "reverse line movement" (public heavily on one side, but line moves the other way)
- Multi-bookmaker comparison: track Pinnacle, Betfair, bet365 spread simultaneously

---

## Priority 2 — Team/Player Intelligence

### Player Impact Model
Absence-weighted player value scoring. Calculates impact of missing players:
- Striker absent → reduce expected goals by 10–20%
- First-choice keeper absent → increase goals conceded by 15–25%
- Defender (CB or fullback) absent → increase goals conceded by 5–12%
- Playmaker absent → reduce xG by 8–15%

Values calibrated from historical outcomes when player was absent vs present.

### Team Style Matchup Engine
- High press vs low block: possession-based teams vs defensive teams
- Set-piece danger: dead-ball specialists vs weak aerial defences
- Counter-attack vulnerability: high-line teams vs fast strikers
- Implementation: cluster teams by playing style using last 20 matches' stats

---

## Priority 3 — Match Context

### Motivation/Context Engine
Probability adjustment multipliers based on match context:

| Context | Adjustment |
|---------|-----------|
| Title race (must-win) | +5–10% goal probability |
| Relegation six-pointer | High variance — both outcomes amplified |
| Dead rubber (mid-table, nothing to play for) | –5–10% goal probability |
| Cup final / knockout | Apply standard model — no dead rubber risk |
| Derby match | Increased variance, reduced home advantage |
| Manager under threat | Unpredictable — flag, do not adjust |

---

## Priority 4 — AI Decision Layer Upgrades

### No-Bet Detector
Produces explicit "AVOID" signals, not just absence of a bet signal. Conditions:
- Any `fallback_flag` is True
- Odds outside 1.20–3.00 range
- Model probability and bookmaker implied probability in close disagreement
- Historical CLV for this segment is negative
- More than 2 data quality flags active simultaneously

### Confidence-by-Segment System
- Track model calibration separately by: league, market type, odds bucket, season phase
- Adjust displayed confidence based on segment historical accuracy
- Lower displayed confidence in leagues where calibration is weaker (e.g., Conference League)

### Bet Memory System (Similar-Match Retrieval)
- Store embedding of each prediction packet
- On new prediction, retrieve the 10 most similar historical situations
- Show: "In similar situations (similar edge, similar form differential, same league), the model was correct X% of the time"

### Explainability Layer
- "Why this bet" card: 3–5 bullet points grounding the recommendation in data
- "Why not this bet" card: explicit reasons when PASS/AVOID is issued
- No AI-generated waffle — all reasoning must be derived from the packet

---

## Priority 5 — Quant/Model Science

### Ensemble Expansion
- Add CatBoost and LightGBM alongside XGBoost
- Add market-implied probability model (devigged closing odds as a baseline)
- Weighted average of all four, with weights calibrated per league

### Drift Detection
Automatic monitoring for:
- ROI degradation over rolling 100-bet window
- Calibration drift (Brier score increasing over time)
- CLV degradation (beating closing line less frequently)
- Feature drift (historical stats drifting from current season distribution)

Trigger: if any metric degrades beyond threshold → pause bet recommendations and flag for review.

---

## Priority 6 — Product

### Daily Report
- Best 5 bets of the day (highest AI score, eligible only)
- Safest 3 bets (lowest variance, highest confidence)
- Value picks (highest edge, even if lower confidence)
- No-bet warnings (specific matches to avoid, with reasons)

### User Modes
| Mode | Description |
|------|-------------|
| Conservative | Only STRONG BET recommendations, tight odds range |
| Balanced | BET + SMALL BET, standard filters |
| Aggressive | All eligible predictions, wider edge tolerance |

### Notifications
- Telegram/Discord: daily best bets, strong bet alerts, odds movement alerts
- Email digest: weekly ROI summary, CLV performance, segment analysis
