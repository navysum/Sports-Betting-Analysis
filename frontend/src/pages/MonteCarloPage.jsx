import { useState, useEffect, useCallback } from "react";
import { getTodayPredictions } from "../services/api";
import { kickoffTime } from "../utils/time";

// ─── Simulation constants ─────────────────────────────────────────────────────

const N_SIMS         = 100_000;
const NB_DISPERSION  = 8;      // Negative Binomial dispersion r — variance = μ + μ²/r
const PARAM_SIGMA    = 0.12;   // Log-scale std dev for parameter uncertainty
const PARAM_BATCH    = 2_000;  // Re-draw λ from uncertainty dist every N sims
const MAX_DISPLAY    = 6;      // Max goals shown in heatmap axes

// ─── Sampling primitives ──────────────────────────────────────────────────────

/** Standard normal via Box-Muller transform */
function normalSample() {
  const u1 = Math.random(), u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Gamma(shape, 1) via Marsaglia-Tsang (2000).
 * Correct for shape ≥ 1; shape < 1 handled via boost trick.
 */
function gammaSample(shape) {
  if (shape < 1) {
    return gammaSample(shape + 1) * Math.pow(Math.random() + 1e-15, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = normalSample(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u + 1e-15) < 0.5 * x * x + d * (1 - v + Math.log(v + 1e-15))) return d * v;
  }
}

/**
 * Negative Binomial(μ, r) via Poisson-Gamma mixture.
 * When r → ∞ collapses to Poisson(μ).
 * Football overdispersion: variance ≈ μ + μ²/r, r ≈ 8 per team.
 */
function negBinSample(mu, r = NB_DISPERSION) {
  if (mu <= 0) return 0;
  // G ~ Gamma(r, μ/r) → mean=μ, then X | G ~ Poisson(G)
  const g = gammaSample(r) * (mu / r);
  return poissonSample(g);
}

/** Poisson(λ) — Knuth algorithm. Kept as utility. */
function poissonSample(lambda) {
  if (lambda <= 0) return 0;
  if (lambda > 30) {
    // For large λ use normal approximation to avoid floating underflow
    return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * normalSample()));
  }
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= Math.random(); } while (p > L);
  return k - 1;
}

/**
 * Draw λ from LogNormal(log(λ̂), σ) — models our uncertainty in the expected
 * goals estimate. σ = 0.12 means the true λ could plausibly be ±12% (log scale).
 */
function perturbLambda(lambda, sigma = PARAM_SIGMA) {
  return Math.exp(Math.log(lambda + 1e-6) + sigma * normalSample());
}

// ─── DC score-grid sampling ───────────────────────────────────────────────────

/**
 * Build a flat CDF from the Dixon-Coles score grid for fast inversion sampling.
 * The grid is τ-corrected (low-score probabilities are properly calibrated),
 * making this far more accurate than raw Poisson for correct-score markets.
 */
function buildScoreCDF(scoreGrid) {
  if (!scoreGrid || !scoreGrid.length) return null;
  const cdf = [];
  let cumsum = 0;
  for (let h = 0; h < scoreGrid.length; h++) {
    for (let a = 0; a < scoreGrid[h].length; a++) {
      cumsum += scoreGrid[h][a];
      cdf.push({ h, a, c: cumsum });
    }
  }
  return cdf;
}

/** Sample (homeGoals, awayGoals) from a pre-built DC CDF via binary search. */
function sampleFromCDF(cdf) {
  const r = Math.random();
  let lo = 0, hi = cdf.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cdf[mid].c < r) lo = mid + 1;
    else hi = mid;
  }
  return [cdf[lo].h, cdf[lo].a];
}

// ─── Main simulation engine ───────────────────────────────────────────────────

/**
 * Run N Monte Carlo simulations and return a comprehensive probability breakdown.
 *
 * Sampling priority:
 *   1. DC corrected score grid — τ-corrected bivariate distribution (best)
 *   2. Negative Binomial — if no DC grid, captures goal overdispersion
 *
 * Parameter uncertainty: every PARAM_BATCH sims, re-draw λH and λA from a
 * log-normal distribution centred on the point estimate. This widens confidence
 * intervals to reflect model uncertainty, not just sampling variance.
 *
 * @param {number}   lambdaHome    - Expected goals home (DC point estimate)
 * @param {number}   lambdaAway    - Expected goals away
 * @param {Array}    scoreGrid     - Full DC τ-corrected probability grid (or null)
 * @param {object}   bookmakerOdds - {home, draw, away} decimal odds (or null)
 * @param {number}   N             - Number of simulations
 */
function runSimulation(lambdaHome, lambdaAway, scoreGrid = null, bookmakerOdds = null, N = N_SIMS) {
  const cdf = buildScoreCDF(scoreGrid);

  // Outcome counters
  let homeWins = 0, draws = 0, awayWins = 0;
  let over15 = 0, over25 = 0, over35 = 0, over45 = 0, btts = 0;
  const goalTotals = new Array(9).fill(0);
  const scoreMap = {};

  // Conditional market accumulators
  let bttsAndHomeWin = 0, bttsAndDraw = 0, bttsAndAwayWin = 0;
  let over25AndDraw = 0, over25AndHomeWin = 0;
  let cleanSheetHome = 0, cleanSheetAway = 0;

  // Asian Handicap: track goal diff
  // ah[line] = {win, push, lose}  (from HOME team's perspective)
  const AH_LINES = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5];
  const ah = {};
  AH_LINES.forEach(l => { ah[l] = { win: 0, push: 0, lose: 0 }; });

  // Goal margin distribution (for draw no-bet, 3-way handicap)
  const marginCounts = {};

  // Convergence: snapshot every 1000 sims
  const convergence = [];

  let lH = lambdaHome, lA = lambdaAway;

  for (let i = 0; i < N; i++) {
    // Re-draw λ from uncertainty distribution every batch
    if (i % PARAM_BATCH === 0) {
      lH = perturbLambda(lambdaHome);
      lA = perturbLambda(lambdaAway);
    }

    let h, a;
    if (cdf) {
      // Primary: sample directly from DC τ-corrected grid
      [h, a] = sampleFromCDF(cdf);
    } else {
      // Fallback: Negative Binomial (overdispersed Poisson)
      h = negBinSample(lH);
      a = negBinSample(lA);
    }

    // ── Result ───────────────────────────────────────────────────────────────
    const diff = h - a;
    const total = h + a;
    const isBTTS = h > 0 && a > 0;

    if (diff > 0) {
      homeWins++;
      if (isBTTS) bttsAndHomeWin++;
      if (total > 2) over25AndHomeWin++;
    } else if (diff < 0) {
      awayWins++;
      if (isBTTS) bttsAndAwayWin++;
    } else {
      draws++;
      if (isBTTS) bttsAndDraw++;
      if (total > 2) over25AndDraw++;
    }
    if (isBTTS) btts++;
    if (a === 0) cleanSheetHome++;
    if (h === 0) cleanSheetAway++;

    // ── Goals markets ─────────────────────────────────────────────────────────
    if (total > 1) over15++;
    if (total > 2) over25++;
    if (total > 3) over35++;
    if (total > 4) over45++;
    goalTotals[Math.min(total, 8)]++;

    // ── Score map ─────────────────────────────────────────────────────────────
    const key = `${Math.min(h, MAX_DISPLAY)}-${Math.min(a, MAX_DISPLAY)}`;
    scoreMap[key] = (scoreMap[key] || 0) + 1;

    // ── Margin distribution ───────────────────────────────────────────────────
    marginCounts[diff] = (marginCounts[diff] || 0) + 1;

    // ── Asian Handicap (from HOME perspective) ────────────────────────────────
    // Handicap h applied to home: adjusted diff = diff + h
    // Line -1.5: home wins AH if diff >= 2 (no push possible with .5)
    // Line -1.0: home wins AH if diff >= 2, push if diff == 1, lose if diff <= 0
    // Line -0.5: home wins AH if diff >= 1
    // Line +0.5: home wins AH if diff >= 0 (home win or draw)
    // Line +1.0: home wins AH if diff >= 0, push if diff == -1
    // Line +1.5: home wins AH if diff >= -1
    for (const line of AH_LINES) {
      const adj = diff - line; // adjusted difference (positive = home covers)
      if (Math.abs(line % 1) > 0) {
        // Half-line: no push possible
        if (adj > 0) ah[line].win++;
        else         ah[line].lose++;
      } else {
        // Whole-line: push when exactly 0
        if (adj > 0)      ah[line].win++;
        else if (adj < 0) ah[line].lose++;
        else              ah[line].push++;
      }
    }

    // ── Convergence snapshot ──────────────────────────────────────────────────
    if ((i + 1) % 1000 === 0) {
      convergence.push({ n: i + 1, pct: (homeWins / (i + 1)) * 100 });
    }
  }

  // ── Derived percentages ───────────────────────────────────────────────────

  const homeWinPct = (homeWins / N) * 100;
  const drawPct    = (draws    / N) * 100;
  const awayWinPct = (awayWins / N) * 100;

  // Confidence intervals: combine sampling variance + parameter uncertainty.
  // Extra sigma from parameter uncertainty: approximately PARAM_SIGMA × √(p(1-p))
  function ci(count) {
    const p = count / N;
    const samplingVar = p * (1 - p) / N;
    const modelVar    = (PARAM_SIGMA * 0.5) ** 2 * p * (1 - p); // approximate
    return 1.96 * Math.sqrt(samplingVar + modelVar) * 100;
  }

  // ── Asian Handicap return probability ─────────────────────────────────────
  // AH return = win% + push% × 0.5 (push refunds half stake at fair odds)
  const ahStats = {};
  AH_LINES.forEach(l => {
    const { win, push, lose } = ah[l];
    ahStats[l] = {
      win:    round2(win    / N * 100),
      push:   round2(push   / N * 100),
      lose:   round2(lose   / N * 100),
      // Effective win rate (treating push as half-win, relevant for fair-odds EV)
      eff:    round2((win + push * 0.5) / N * 100),
    };
  });

  // ── Expected Value vs bookmaker odds ─────────────────────────────────────
  let ev = null;
  if (bookmakerOdds) {
    const { home: ho, draw: do_, away: ao,
            over25: ou25, over35: ou35, btts: bto } = bookmakerOdds;
    ev = {
      home:   ho   ? round3((homeWins / N) * ho - 1)       : null,
      draw:   do_  ? round3((draws    / N) * do_ - 1)      : null,
      away:   ao   ? round3((awayWins / N) * ao - 1)       : null,
      over25: ou25 ? round3((over25   / N) * ou25 - 1)     : null,
      over35: ou35 ? round3((over35   / N) * ou35 - 1)     : null,
      btts:   bto  ? round3((btts     / N) * bto - 1)      : null,
    };
  }

  // ── Conditional markets ───────────────────────────────────────────────────
  const conditionals = {
    bttsGivenHome:      homeWins > 0 ? round2(bttsAndHomeWin / homeWins * 100) : 0,
    bttsGivenDraw:      draws > 0    ? round2(bttsAndDraw    / draws    * 100) : 0,
    bttsGivenAway:      awayWins > 0 ? round2(bttsAndAwayWin / awayWins * 100) : 0,
    over25GivenDraw:    draws > 0    ? round2(over25AndDraw  / draws    * 100) : 0,
    over25GivenHomeWin: homeWins > 0 ? round2(over25AndHomeWin / homeWins * 100) : 0,
    cleanSheetHome:     round2(cleanSheetHome / N * 100),
    cleanSheetAway:     round2(cleanSheetAway / N * 100),
  };

  // ── Top scorelines ─────────────────────────────────────────────────────────
  const topScores = Object.entries(scoreMap)
    .map(([score, count]) => ({ score, pct: round2((count / N) * 100) }))
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 12);

  return {
    N,
    usingDCGrid: !!cdf,
    homeWinPct: round2(homeWinPct),
    drawPct:    round2(drawPct),
    awayWinPct: round2(awayWinPct),
    homeCI: round2(ci(homeWins)), drawCI: round2(ci(draws)), awayCI: round2(ci(awayWins)),
    over15Pct: round2((over15 / N) * 100),
    over25Pct: round2((over25 / N) * 100),
    over35Pct: round2((over35 / N) * 100),
    over45Pct: round2((over45 / N) * 100),
    bttsPct:   round2((btts   / N) * 100),
    goalTotals: goalTotals.map((c, i) => ({
      label: i === 8 ? "8+" : String(i),
      pct:   round2((c / N) * 100),
    })),
    topScores,
    scoreMap,
    convergence,
    ah: ahStats,
    ev,
    conditionals,
  };
}

function round2(v) { return Math.round(v * 100) / 100; }
function round3(v) { return Math.round(v * 1000) / 1000; }

// ─── Visual components ────────────────────────────────────────────────────────

/** EV badge — green if positive, red if negative */
function EVBadge({ ev }) {
  if (ev == null) return null;
  const pct = (ev * 100).toFixed(1);
  const positive = ev > 0;
  return (
    <span className={`text-[10px] font-mono px-1 py-0.5 rounded tabular-nums
                      ${positive ? "bg-green-900/40 text-green-400" : "bg-red-900/30 text-red-500"}`}>
      {positive ? "+" : ""}{pct}%
    </span>
  );
}

/** Horizontal bar with CI whisker and optional EV badge */
function ResultBar({ label, pct, ciPct, color, highlight, evVal }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`w-12 shrink-0 ${highlight ? "text-zinc-200" : "text-zinc-500"}`}>{label}</span>
      <div className="flex-1 h-2.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
      <span className="text-zinc-400 tabular-nums w-12 text-right">
        {pct.toFixed(1)}%
        {ciPct != null && (
          <span className="text-zinc-600 ml-0.5 text-[9px]">±{ciPct.toFixed(1)}</span>
        )}
      </span>
      <div className="w-14 text-right">
        <EVBadge ev={evVal} />
      </div>
    </div>
  );
}

/** Horizontal bar for goals / markets */
function MarketBar({ label, pct, color, evVal }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`w-12 shrink-0 ${pct >= 50 ? "text-zinc-200" : "text-zinc-500"}`}>{label}</span>
      <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
      <span className="text-zinc-500 tabular-nums w-10 text-right">{pct.toFixed(1)}%</span>
      <div className="w-14 text-right">
        <EVBadge ev={evVal} />
      </div>
    </div>
  );
}

/** Goal total histogram */
function GoalTotalChart({ goalTotals }) {
  const max = Math.max(...goalTotals.map(g => g.pct), 0.01);
  return (
    <div className="space-y-1">
      {goalTotals.map(({ label, pct }) => (
        <div key={label} className="flex items-center gap-2 text-xs">
          <span className="text-zinc-500 w-4 text-right tabular-nums">{label}</span>
          <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-indigo-500"
                 style={{ width: `${(pct / max) * 100}%` }} />
          </div>
          <span className="text-zinc-500 tabular-nums w-8 text-right">{pct.toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

/** Asian Handicap table */
function AsianHandicapTable({ ah, homeLabel, awayLabel }) {
  const lines = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5];
  return (
    <div className="overflow-x-auto">
      <table className="text-[10px] w-full">
        <thead>
          <tr className="text-zinc-600">
            <th className="text-left py-1 pr-2 font-normal">Line</th>
            <th className="text-right py-1 px-1 font-normal">{homeLabel.split(" ").pop()} Win%</th>
            {lines.some(l => l % 1 === 0) && (
              <th className="text-right py-1 px-1 font-normal text-zinc-700">Push%</th>
            )}
            <th className="text-right py-1 px-1 font-normal">{awayLabel.split(" ").pop()} Win%</th>
            <th className="text-right py-1 pl-2 font-normal text-zinc-600">Eff%</th>
          </tr>
        </thead>
        <tbody>
          {lines.map(line => {
            const s = ah[line];
            if (!s) return null;
            const isHomeAdv = line < 0;
            const effClass = s.eff > 50 ? "text-green-400" : s.eff < 50 ? "text-zinc-500" : "text-zinc-400";
            return (
              <tr key={line} className="border-t border-zinc-800/40">
                <td className={`py-1 pr-2 font-mono ${isHomeAdv ? "text-zinc-300" : "text-zinc-500"}`}>
                  {line > 0 ? `+${line}` : line}
                </td>
                <td className={`text-right px-1 tabular-nums ${isHomeAdv ? "text-zinc-300" : "text-zinc-600"}`}>
                  {s.win.toFixed(1)}
                </td>
                {s.push > 0 && (
                  <td className="text-right px-1 tabular-nums text-zinc-700">{s.push.toFixed(1)}</td>
                )}
                {s.push === 0 && lines.some(l => l % 1 === 0) && (
                  <td className="text-right px-1 text-zinc-800">—</td>
                )}
                <td className={`text-right px-1 tabular-nums ${!isHomeAdv ? "text-zinc-300" : "text-zinc-600"}`}>
                  {s.lose.toFixed(1)}
                </td>
                <td className={`text-right pl-2 tabular-nums font-medium ${effClass}`}>
                  {s.eff.toFixed(1)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="text-[9px] text-zinc-700 mt-1">
        Eff% = win% + push%×0.5 from home team perspective · line = home handicap
      </p>
    </div>
  );
}

/** Conditional markets panel */
function ConditionalMarkets({ cond, homeLabel, awayLabel }) {
  const rows = [
    { label: `BTTS if ${homeLabel.split(" ").pop()} win`, val: cond.bttsGivenHome, tip: "Quality home wins (not clean sheets)" },
    { label: `BTTS if Draw`,                              val: cond.bttsGivenDraw, tip: "Scoring draws (1-1, 2-2…) vs 0-0" },
    { label: `BTTS if ${awayLabel.split(" ").pop()} win`, val: cond.bttsGivenAway, tip: "Quality away wins vs clean sheets" },
    { label: `O 2.5 if Draw`,                             val: cond.over25GivenDraw, tip: "High-scoring draws (2-2, 2-1 excl.)" },
    { label: `O 2.5 if Home Win`,                         val: cond.over25GivenHomeWin },
    { label: `${homeLabel.split(" ").pop()} clean sheet`, val: cond.cleanSheetHome },
    { label: `${awayLabel.split(" ").pop()} clean sheet`, val: cond.cleanSheetAway },
  ];
  return (
    <div className="space-y-1.5">
      {rows.map(({ label, val }) => (
        <div key={label} className="flex items-center gap-2 text-xs">
          <span className="text-zinc-600 flex-1 min-w-0 truncate">{label}</span>
          <div className="w-24 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-violet-500"
                 style={{ width: `${Math.min(val, 100)}%` }} />
          </div>
          <span className={`tabular-nums w-10 text-right text-xs ${val >= 50 ? "text-zinc-300" : "text-zinc-500"}`}>
            {val.toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

/** Score heatmap — 0 to MAX_DISPLAY goals per side */
function ScoreHeatmap({ scoreMap, N, homeLabel, awayLabel }) {
  const MAX = MAX_DISPLAY;
  let peak = 0, topPct = 0;
  for (let h = 0; h <= MAX; h++) {
    for (let a = 0; a <= MAX; a++) {
      const c = scoreMap[`${h}-${a}`] || 0;
      if (c > peak) { peak = c; topPct = (c / N) * 100; }
    }
  }
  return (
    <div className="overflow-x-auto">
      <p className="text-[9px] text-zinc-600 mb-1 ml-7">{awayLabel} goals →</p>
      <div className="flex gap-0.5 mb-0.5 ml-7">
        {Array.from({ length: MAX + 1 }, (_, i) => (
          <div key={i} className="w-7 text-center text-[9px] text-zinc-600">{i}</div>
        ))}
      </div>
      {Array.from({ length: MAX + 1 }, (_, h) => (
        <div key={h} className="flex items-center gap-0.5 mb-0.5">
          <div className="w-6 text-[9px] text-zinc-600 text-right pr-1">{h}</div>
          {Array.from({ length: MAX + 1 }, (_, a) => {
            const c = scoreMap[`${h}-${a}`] || 0;
            const pct = (c / N) * 100;
            const intensity = peak > 0 ? c / peak : 0;
            const isTop = pct === topPct;
            return (
              <div key={a}
                   title={`${h}-${a}: ${pct.toFixed(2)}%`}
                   className={`w-7 h-7 rounded flex items-center justify-center text-[9px] tabular-nums
                               ${isTop ? "ring-1 ring-green-500/60 font-bold text-white" : "text-zinc-500"}`}
                   style={{ backgroundColor: `rgba(34,197,94,${0.04 + intensity * 0.58})` }}>
                {pct >= 1 ? `${Math.round(pct)}%` : ""}
              </div>
            );
          })}
        </div>
      ))}
      <p className="text-[9px] text-zinc-600 mt-1 ml-7">↑ {homeLabel} goals</p>
    </div>
  );
}

/** Convergence line chart */
function ConvergenceChart({ data, finalPct, homeLabel }) {
  if (!data?.length) return null;
  const W = 280, H = 64, PAD = 4;
  const xs = data.map(d => ((d.n - 1000) / (N_SIMS - 1000)) * (W - PAD * 2) + PAD);
  const minY = Math.min(...data.map(d => d.pct), finalPct) - 4;
  const maxY = Math.max(...data.map(d => d.pct), finalPct) + 4;
  const toY = pct => H - PAD - ((pct - minY) / (maxY - minY || 1)) * (H - PAD * 2);
  const pts = data.map((d, i) => `${xs[i]},${toY(d.pct)}`).join(" ");
  const area = `M${xs[0]},${H} ` + data.map((d, i) => `L${xs[i]},${toY(d.pct)}`).join(" ")
             + ` L${xs[xs.length - 1]},${H} Z`;
  return (
    <div>
      <p className="text-[9px] text-zinc-600 mb-1">
        {homeLabel} win % over {N_SIMS.toLocaleString()} simulations
      </p>
      <svg width={W} height={H} className="overflow-visible">
        <line x1={PAD} y1={toY(finalPct)} x2={W - PAD} y2={toY(finalPct)}
              stroke="#22c55e" strokeWidth={0.5} strokeDasharray="3,3" opacity={0.4} />
        <path d={area} fill="rgba(34,197,94,0.06)" />
        <polyline points={pts} fill="none" stroke="#22c55e" strokeWidth={1.5} opacity={0.8} />
        <circle cx={xs[xs.length - 1]} cy={toY(data[data.length - 1].pct)} r={2.5} fill="#22c55e" />
        <text x={W - PAD + 3} y={toY(finalPct) + 4} fontSize={8} fill="#22c55e" opacity={0.7}>
          {finalPct.toFixed(1)}%
        </text>
      </svg>
    </div>
  );
}

// ─── Per-match simulation panel ───────────────────────────────────────────────

function SimPanel({ item }) {
  const pred      = item.prediction || {};
  const lambdaH   = pred.xg_home;
  const lambdaA   = pred.xg_away;
  const scoreGrid = pred.score_grid  || null;
  const bookOdds  = pred.bookmaker_odds || null;

  const [sim, setSim]   = useState(null);
  const [seed, setSeed] = useState(0);
  const rerun = useCallback(() => setSeed(s => s + 1), []);

  useEffect(() => {
    if (!lambdaH || !lambdaA) return;
    setSim(null);
    const id = setTimeout(() => {
      setSim(runSimulation(lambdaH, lambdaA, scoreGrid, bookOdds));
    }, 16);
    return () => clearTimeout(id);
  }, [lambdaH, lambdaA, scoreGrid, bookOdds, seed]);

  if (!lambdaH || !lambdaA) {
    return (
      <p className="text-xs text-zinc-600">
        No xG data — Dixon-Coles model unavailable for this match.
      </p>
    );
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[10px] bg-zinc-800 text-zinc-400 rounded px-2 py-0.5 tabular-nums">
            {N_SIMS.toLocaleString()} simulations
          </span>
          <span className="text-[10px] text-zinc-600">
            λ = {lambdaH.toFixed(2)} / {lambdaA.toFixed(2)}
          </span>
          {sim && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded tabular-nums
                              ${sim.usingDCGrid
                                ? "bg-green-900/30 text-green-600"
                                : "bg-zinc-800 text-zinc-500"}`}>
              {sim.usingDCGrid ? "DC τ-corrected" : "Neg. Binomial fallback"}
            </span>
          )}
        </div>
        <button
          onClick={rerun}
          className="text-[10px] text-zinc-500 hover:text-zinc-200 border border-zinc-700 rounded px-2 py-0.5 transition-colors"
        >
          Re-run
        </button>
      </div>

      {!sim && <p className="text-xs text-zinc-600 animate-pulse">Running {N_SIMS.toLocaleString()} sims…</p>}

      {sim && (
        <>
          {/* ── Result probabilities ── */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Result Probability
              {bookOdds && <span className="text-zinc-700 normal-case ml-2 tracking-normal">· EV vs bookmaker</span>}
            </p>
            <div className="space-y-1.5">
              <ResultBar label={item.home_team.split(" ").pop()} pct={sim.homeWinPct}
                         ciPct={sim.homeCI} color="bg-green-500"
                         highlight={sim.homeWinPct > sim.drawPct && sim.homeWinPct > sim.awayWinPct}
                         evVal={sim.ev?.home} />
              <ResultBar label="Draw" pct={sim.drawPct}
                         ciPct={sim.drawCI} color="bg-yellow-500"
                         highlight={sim.drawPct > sim.homeWinPct && sim.drawPct > sim.awayWinPct}
                         evVal={sim.ev?.draw} />
              <ResultBar label={item.away_team.split(" ").pop()} pct={sim.awayWinPct}
                         ciPct={sim.awayCI} color="bg-blue-500"
                         highlight={sim.awayWinPct > sim.homeWinPct && sim.awayWinPct > sim.drawPct}
                         evVal={sim.ev?.away} />
            </div>
            <p className="text-[9px] text-zinc-700 mt-1.5">
              ± includes sampling variance + λ estimation uncertainty (σ={PARAM_SIGMA})
            </p>
          </div>

          {/* ── Goals markets ── */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">Goals Markets</p>
            <div className="space-y-1.5">
              {[
                { label: "O 1.5", pct: sim.over15Pct, color: "bg-sky-500",    evVal: null },
                { label: "O 2.5", pct: sim.over25Pct, color: "bg-blue-500",   evVal: sim.ev?.over25 },
                { label: "O 3.5", pct: sim.over35Pct, color: "bg-indigo-500", evVal: sim.ev?.over35 },
                { label: "O 4.5", pct: sim.over45Pct, color: "bg-violet-500", evVal: null },
                { label: "U 2.5", pct: 100 - sim.over25Pct, color: "bg-orange-500", evVal: null },
                { label: "BTTS",  pct: sim.bttsPct,   color: "bg-purple-500", evVal: sim.ev?.btts },
              ].map(({ label, pct, color, evVal }) => (
                <MarketBar key={label} label={label} pct={pct} color={color} evVal={evVal} />
              ))}
            </div>
          </div>

          {/* ── Asian Handicap ── */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Asian Handicap
            </p>
            <AsianHandicapTable
              ah={sim.ah}
              homeLabel={item.home_team}
              awayLabel={item.away_team}
            />
          </div>

          {/* ── Conditional markets ── */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Conditional Markets
            </p>
            <ConditionalMarkets
              cond={sim.conditionals}
              homeLabel={item.home_team}
              awayLabel={item.away_team}
            />
          </div>

          {/* ── Total goals distribution ── */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Total Goals Distribution
            </p>
            <GoalTotalChart goalTotals={sim.goalTotals} />
          </div>

          {/* ── Score heatmap ── */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Score Heatmap
            </p>
            <ScoreHeatmap
              scoreMap={sim.scoreMap}
              N={sim.N}
              homeLabel={item.home_team}
              awayLabel={item.away_team}
            />
          </div>

          {/* ── Top scorelines ── */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Most Likely Scorelines
            </p>
            <div className="flex flex-wrap gap-2">
              {sim.topScores.map((s, i) => {
                const maxPct    = sim.topScores[0]?.pct || 1;
                const intensity = s.pct / maxPct;
                return (
                  <div key={s.score}
                       className={`flex flex-col items-center justify-center rounded border px-2.5 py-1.5
                                   ${i === 0 ? "border-green-500/50" : "border-zinc-700"}`}
                       style={{ backgroundColor: `rgba(34,197,94,${0.03 + intensity * 0.15})` }}>
                    <span className="font-mono text-sm text-zinc-200">{s.score}</span>
                    <span className="text-[10px] text-zinc-500">{s.pct.toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ── Convergence chart ── */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">Convergence</p>
            <ConvergenceChart data={sim.convergence} finalPct={sim.homeWinPct} homeLabel={item.home_team} />
          </div>
        </>
      )}
    </div>
  );
}

// ─── Expandable match row ─────────────────────────────────────────────────────

function MatchSection({ item }) {
  const [open, setOpen] = useState(false);
  const pred    = item.prediction || {};
  const hasData = pred.xg_home != null;

  return (
    <div className="border-b border-zinc-800/60 last:border-0">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-zinc-900/40 transition-colors"
      >
        <span className="text-xs text-zinc-600 tabular-nums w-10 shrink-0">
          {kickoffTime(item.match_date)}
        </span>
        <div className="flex items-center gap-1.5 flex-1 min-w-0">
          {item.home_team_crest && (
            <img src={item.home_team_crest} alt="" className="w-4 h-4 object-contain opacity-70 shrink-0" />
          )}
          <span className="text-sm truncate">{item.home_team}</span>
        </div>
        <span className="text-xs text-zinc-700 shrink-0">—</span>
        <div className="flex items-center gap-1.5 flex-1 min-w-0 justify-end">
          <span className="text-sm truncate text-right">{item.away_team}</span>
          {item.away_team_crest && (
            <img src={item.away_team_crest} alt="" className="w-4 h-4 object-contain opacity-70 shrink-0" />
          )}
        </div>
        {hasData ? (
          <span className="text-[10px] text-zinc-600 shrink-0 w-14 text-right">
            λ {pred.xg_home?.toFixed(1)}–{pred.xg_away?.toFixed(1)}
          </span>
        ) : (
          <span className="w-14 shrink-0" />
        )}
        <svg className={`w-3.5 h-3.5 text-zinc-600 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
             fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="px-4 pb-6 space-y-5 border-t border-zinc-800/40 pt-4">
          <SimPanel item={item} />
        </div>
      )}
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function MonteCarloPage() {
  const [status, setStatus] = useState("idle");
  const [items, setItems]   = useState([]);

  const dateLabel = new Date().toLocaleDateString("en-GB", {
    weekday: "long", day: "numeric", month: "long",
  });

  useEffect(() => {
    async function load() {
      try {
        const res = await getTodayPredictions();
        const d = res.data;
        setStatus(d.status);
        setItems(d.predictions || []);
      } catch {
        setStatus("error");
      }
    }
    load();
  }, []);

  // xg_home is always set by the backend (DC lookup or over25-derived fallback)
  const sorted = [...items]
    .filter(i => i.prediction != null)
    .sort((a, b) => (a.match_date || "").localeCompare(b.match_date || ""));

  return (
    <div className="max-w-3xl mx-auto content-pad">
      <div className="px-4 pt-5 pb-3">
        <h1 className="text-base font-semibold text-white">Monte Carlo Sim</h1>
        <p className="text-xs text-zinc-600 mt-0.5">
          {dateLabel} · {N_SIMS.toLocaleString()} sims · DC τ-corrected grid · NB overdispersion · parameter uncertainty
        </p>
      </div>

      {status === "idle"      && <div className="px-4 py-12 text-center text-xs text-zinc-600">Loading…</div>}
      {status === "computing" && <div className="px-4 py-3 text-xs text-zinc-600">Predictions computing — showing available data</div>}
      {status === "error"     && <div className="px-4 py-3 text-xs text-red-500">Failed to load. Is the backend running?</div>}
      {(status === "ready" || status === "computing") && sorted.length === 0 && (
        <div className="px-4 py-12 text-center text-xs text-zinc-600">
          No matches scheduled today in supported competitions
        </div>
      )}

      {sorted.length > 0 && (
        <div className="bg-zinc-900/30">
          {sorted.map(item => (
            <MatchSection key={item.api_match_id} item={item} />
          ))}
        </div>
      )}
    </div>
  );
}
