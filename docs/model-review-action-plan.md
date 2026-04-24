# Sports Betting Analysis — Model Review & Action Plan

**Author:** Craig Ataide  
**Date:** April 2026

---

## Verdict

The model is a serious, well-structured framework — above average for an independent football betting model. It is **not** yet proven to have a durable live betting edge.

The architecture (Dixon-Coles + XGBoost ensemble, calibration, devigging, CLV tracking, Kelly sizing, temporal CV) is the shape of a professional betting system. The weakness is a lack of rigorous out-of-sample evidence that the edge is real and stable.

---

## Strengths

- **Temporal leakage awareness** — the codebase explicitly guards against future data contaminating training folds via `TimeSeriesSplit`
- **Probability-first design** — the model generates calibrated probabilities, deviggs bookmaker prices, and only flags bets when model probability exceeds fair implied probability by a threshold
- **Fallback quality tracking** — every prediction carries `fallback_flags` indicating whether proxies were used, so quality is always known
- **Bet eligibility separation** — "show prediction" and "show as bet recommendation" are explicit separate decisions
- **CLV tracking** — closing line value is logged for every prediction, enabling retrospective edge proof

---

## Critical Weaknesses

### 1. No proof of durable live edge
No stored benchmark output, no season-by-season ROI breakdown, no rolling CLV distribution over meaningful sample size.

### 2. Feature set weaker than the market in top leagues
Missing:
- Projected lineups (starting XI vs B team)
- Player-level absence weighting (key striker out = material probability shift)
- Manager effects and tactical matchup features
- Market movement features (steam moves, reverse line movement)

### 3. Training xG ≠ live xG
Training features use shots-on-target × 0.27 as an xG proxy. Live inference uses Understat or Sofascore actual xG values. This feature-distribution mismatch degrades model calibration for xG-heavy predictions.

### 4. Silent fallbacks erode live edge without detection
When the model falls back to global parameters (no DC convergence, no league-specific model, no real xG), the prediction still displays. This is controlled by `fallback_flags` but the end user may not notice.

### 5. Wrong success metric
A 75%+ win rate is **not** a valid broad target. A model can win 80% of bets simply by taking short-priced favourites and still lose money.

**The right metrics:**
- Positive expected value (EV)
- Positive out-of-sample ROI by segment
- Closing Line Value (CLV) > 0 across a meaningful sample

---

## Phase 1 — Immediate Actions

These are implemented or in progress in this codebase:

| # | Action | Status |
|---|--------|--------|
| 1 | Full evaluation report (ROI by market / league / season / bucket) | Partial — backtest endpoint exists |
| 2 | Fallback flags on every prediction | ✅ Done |
| 3 | Bet eligibility gate | ✅ Done |
| 4 | Odds-band and market-band filtering | ✅ Done (via edge + confidence thresholds) |
| 5 | CLV summary output to logs | ✅ Done |
| 6 | Calibration plots | Partial — backtest Brier scores exist |
| 7 | Split prediction display from bet recommendation | ✅ Done |

---

## Bet Eligibility Gate

The gate implemented in `backend/ml/predict.py`:

```python
bet_eligible = all([
    not used_approx_devig,    # exact two-sided devig required
    not used_xg_fallback,     # real xG source, not shots proxy
    not used_dc_fallback,     # Dixon-Coles converged for this fixture
    not used_global_model,    # league-specific model active
    best_edge >= 0.05,        # ≥ 5% edge over fair implied probability
    confidence >= 0.55,       # ≥ 55% model confidence on predicted outcome
])
```

**Only `bet_eligible = True` predictions surface on the Best Bets page.**

---

## Selective High-Hit-Rate Rule Set (Version 1)

A tighter filter for highest-confidence recommendations:

- **Leagues:** PL, PD, BL1, ELC only
- **Markets:** Over 1.5, Double Chance, Team Over 0.5, Strong Favourite 1X2
- **Data quality:** exact devig + no xG fallback + no DC fallback
- **Thresholds:** confidence ≥ 0.68, edge ≥ 0.04, odds 1.25–1.70
- **Historical:** segment must show positive historical CLV

---

## Guiding Principles

> "Do not chase 75% overall hit rate as the main metric. Instead: narrow the scope, improve data quality, tighten bet selection, prove CLV, prove ROI by segment, only then scale."

> "A model can win 75% or 80% of bets simply by taking short-priced favourites, and still lose money."

Trust ROI over win rate. Trust CLV over hype. If edge is weak → PASS.
