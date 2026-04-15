import { useState, useEffect, useCallback } from "react";
import { getTodayPredictions } from "../services/api";
import { kickoffTime } from "../utils/time";

const N_SIMS = 100_000;

// ─── Simulation engine ────────────────────────────────────────────────────────

/** Sample from Poisson(λ) — Knuth algorithm */
function poissonSample(lambda) {
  if (lambda <= 0) return 0;
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= Math.random(); } while (p > L);
  return k - 1;
}

function runSimulation(lambdaHome, lambdaAway, N = N_SIMS) {
  let homeWins = 0, draws = 0, awayWins = 0;
  let over15 = 0, over25 = 0, over35 = 0, btts = 0;
  const goalTotals = new Array(8).fill(0);   // 0..6 + 7+
  const scoreMap = {};                        // "h-a" → count

  // For convergence chart — record home-win% every 200 sims
  const convergence = [];

  for (let i = 0; i < N; i++) {
    const h = poissonSample(lambdaHome);
    const a = poissonSample(lambdaAway);

    if (h > a)      homeWins++;
    else if (h < a) awayWins++;
    else            draws++;

    const total = h + a;
    if (total > 1) over15++;
    if (total > 2) over25++;
    if (total > 3) over35++;
    if (h > 0 && a > 0) btts++;
    goalTotals[Math.min(total, 7)]++;

    const key = `${Math.min(h, 6)}-${Math.min(a, 6)}`;
    scoreMap[key] = (scoreMap[key] || 0) + 1;

    if ((i + 1) % 1000 === 0) {
      convergence.push({ n: i + 1, pct: (homeWins / (i + 1)) * 100 });
    }
  }

  // 95 % CI: ±1.96 * sqrt(p*(1-p)/N)
  function ci(count) {
    const p = count / N;
    return 1.96 * Math.sqrt((p * (1 - p)) / N) * 100;
  }

  const topScores = Object.entries(scoreMap)
    .map(([score, count]) => ({ score, pct: (count / N) * 100 }))
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 12);

  return {
    N,
    homeWinPct: (homeWins / N) * 100,
    drawPct:    (draws    / N) * 100,
    awayWinPct: (awayWins / N) * 100,
    homeCI: ci(homeWins), drawCI: ci(draws), awayCI: ci(awayWins),
    over15Pct: (over15 / N) * 100,
    over25Pct: (over25 / N) * 100,
    over35Pct: (over35 / N) * 100,
    bttsPct:   (btts   / N) * 100,
    goalTotals: goalTotals.map((c, i) => ({
      label: i === 7 ? "7+" : String(i),
      pct: (c / N) * 100,
    })),
    topScores,
    scoreMap,
    convergence,
  };
}

// ─── Visual components ────────────────────────────────────────────────────────

/** Horizontal bar with optional CI whisker */
function ResultBar({ label, pct, ciPct, color, highlight }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`w-12 shrink-0 ${highlight ? "text-zinc-200" : "text-zinc-500"}`}>{label}</span>
      <div className="flex-1 h-2.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
      <span className="text-zinc-400 tabular-nums w-16 text-right">
        {pct.toFixed(1)}%
        {ciPct != null && (
          <span className="text-zinc-600 ml-0.5 text-[9px]">±{ciPct.toFixed(1)}</span>
        )}
      </span>
    </div>
  );
}

/** Goal total histogram — horizontal bars */
function GoalTotalChart({ goalTotals }) {
  const max = Math.max(...goalTotals.map(g => g.pct), 0.01);
  return (
    <div className="space-y-1">
      {goalTotals.map(({ label, pct }) => (
        <div key={label} className="flex items-center gap-2 text-xs">
          <span className="text-zinc-500 w-4 text-right tabular-nums">{label}</span>
          <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-indigo-500"
              style={{ width: `${(pct / max) * 100}%` }}
            />
          </div>
          <span className="text-zinc-500 tabular-nums w-8 text-right">{pct.toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

/** 6×6 score heatmap — home goals on Y, away goals on X */
function ScoreHeatmap({ scoreMap, N, homeLabel, awayLabel }) {
  const MAX = 6;
  let peak = 0;
  let topPct = 0;
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
              <div
                key={a}
                title={`${h}-${a}: ${pct.toFixed(1)}%`}
                className={`w-7 h-7 rounded flex items-center justify-center text-[9px] tabular-nums
                            ${isTop ? "ring-1 ring-green-500/60 font-bold text-white" : "text-zinc-500"}`}
                style={{ backgroundColor: `rgba(34,197,94,${0.05 + intensity * 0.55})` }}
              >
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

/** Convergence SVG line chart */
function ConvergenceChart({ data, finalPct, homeLabel }) {
  if (!data?.length) return null;
  const W = 280, H = 64, PAD = 4;
  const xs = data.map(d => ((d.n - 1000) / (N_SIMS - 1000)) * (W - PAD * 2) + PAD);
  const minY = Math.min(...data.map(d => d.pct), finalPct) - 5;
  const maxY = Math.max(...data.map(d => d.pct), finalPct) + 5;
  const toY = pct => H - PAD - ((pct - minY) / (maxY - minY)) * (H - PAD * 2);

  const pts = data.map((d, i) => `${xs[i]},${toY(d.pct)}`).join(" ");
  const area = `M${xs[0]},${H} ` + data.map((d, i) => `L${xs[i]},${toY(d.pct)}`).join(" ") + ` L${xs[xs.length-1]},${H} Z`;

  return (
    <div>
      <p className="text-[9px] text-zinc-600 mb-1">{homeLabel} win % converging over {N_SIMS.toLocaleString()} sims</p>
      <svg width={W} height={H} className="overflow-visible">
        {/* Final value horizontal line */}
        <line x1={PAD} y1={toY(finalPct)} x2={W - PAD} y2={toY(finalPct)}
              stroke="#22c55e" strokeWidth={0.5} strokeDasharray="3,3" opacity={0.4} />
        {/* Area fill */}
        <path d={area} fill="rgba(34,197,94,0.06)" />
        {/* Line */}
        <polyline points={pts} fill="none" stroke="#22c55e" strokeWidth={1.5} opacity={0.8} />
        {/* Final dot */}
        <circle cx={xs[xs.length-1]} cy={toY(data[data.length-1].pct)} r={2.5} fill="#22c55e" />
        {/* Final label */}
        <text x={W - PAD + 3} y={toY(finalPct) + 4} fontSize={8} fill="#22c55e" opacity={0.7}>
          {finalPct.toFixed(1)}%
        </text>
      </svg>
    </div>
  );
}

// ─── Per-match simulation panel ───────────────────────────────────────────────

function SimPanel({ item }) {
  const pred   = item.prediction || {};
  const lambdaH = pred.xg_home;
  const lambdaA = pred.xg_away;
  const [sim, setSim] = useState(null);
  const [seed, setSeed] = useState(0); // increment to re-run

  const rerun = useCallback(() => setSeed(s => s + 1), []);

  useEffect(() => {
    if (!lambdaH || !lambdaA) return;
    // Small timeout so the card animation finishes first
    const id = setTimeout(() => setSim(runSimulation(lambdaH, lambdaA)), 16);
    return () => clearTimeout(id);
  }, [lambdaH, lambdaA, seed]);

  if (!lambdaH || !lambdaA) {
    return <p className="text-xs text-zinc-600">No xG data — Dixon-Coles model unavailable for this match.</p>;
  }

  return (
    <div className="space-y-5">
      {/* Sim header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[10px] bg-zinc-800 text-zinc-400 rounded px-2 py-0.5 tabular-nums">
            {N_SIMS.toLocaleString()} simulations
          </span>
          <span className="text-[10px] text-zinc-600">
            λ = {lambdaH.toFixed(2)} / {lambdaA.toFixed(2)}
          </span>
        </div>
        <button
          onClick={rerun}
          className="text-[10px] text-zinc-500 hover:text-zinc-200 border border-zinc-700 rounded px-2 py-0.5 transition-colors"
        >
          Re-run
        </button>
      </div>

      {!sim && (
        <p className="text-xs text-zinc-600">Running…</p>
      )}

      {sim && (
        <>
          {/* Result probabilities */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Result Probability
            </p>
            <div className="space-y-1.5">
              <ResultBar
                label={item.home_team.split(" ").pop()}
                pct={sim.homeWinPct}
                ciPct={sim.homeCI}
                color="bg-green-500"
                highlight={sim.homeWinPct > sim.drawPct && sim.homeWinPct > sim.awayWinPct}
              />
              <ResultBar
                label="Draw"
                pct={sim.drawPct}
                ciPct={sim.drawCI}
                color="bg-yellow-500"
                highlight={sim.drawPct > sim.homeWinPct && sim.drawPct > sim.awayWinPct}
              />
              <ResultBar
                label={item.away_team.split(" ").pop()}
                pct={sim.awayWinPct}
                ciPct={sim.awayCI}
                color="bg-blue-500"
                highlight={sim.awayWinPct > sim.homeWinPct && sim.awayWinPct > sim.drawPct}
              />
            </div>
            <p className="text-[9px] text-zinc-700 mt-1.5">± values are 95% confidence intervals</p>
          </div>

          {/* Goals market */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Goals Market
            </p>
            <div className="space-y-1.5">
              {[
                { label: "O 1.5", pct: sim.over15Pct, color: "bg-blue-500" },
                { label: "O 2.5", pct: sim.over25Pct, color: "bg-blue-500" },
                { label: "O 3.5", pct: sim.over35Pct, color: "bg-blue-500" },
                { label: "U 2.5", pct: 100 - sim.over25Pct, color: "bg-orange-500" },
                { label: "BTTS",  pct: sim.bttsPct,   color: "bg-purple-500" },
              ].map(({ label, pct, color }) => (
                <div key={label} className="flex items-center gap-2 text-xs">
                  <span className={`w-12 shrink-0 ${pct >= 50 ? "text-zinc-200" : "text-zinc-500"}`}>
                    {label}
                  </span>
                  <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-zinc-500 tabular-nums w-8 text-right">{pct.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Total goals histogram */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Total Goals Distribution
            </p>
            <GoalTotalChart goalTotals={sim.goalTotals} />
          </div>

          {/* Score heatmap */}
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

          {/* Top scorelines */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Most Likely Scorelines
            </p>
            <div className="flex flex-wrap gap-2">
              {sim.topScores.map((s, i) => {
                const maxPct = sim.topScores[0]?.pct || 1;
                const intensity = s.pct / maxPct;
                return (
                  <div
                    key={s.score}
                    className={`flex flex-col items-center justify-center rounded border px-2.5 py-1.5
                                ${i === 0 ? "border-green-500/50" : "border-zinc-700"}`}
                    style={{ backgroundColor: `rgba(34,197,94,${0.03 + intensity * 0.15})` }}
                  >
                    <span className="font-mono text-sm text-zinc-200">{s.score}</span>
                    <span className="text-[10px] text-zinc-500">{s.pct.toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Convergence chart */}
          <div>
            <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
              Convergence
            </p>
            <ConvergenceChart
              data={sim.convergence}
              finalPct={sim.homeWinPct}
              homeLabel={item.home_team}
            />
          </div>
        </>
      )}
    </div>
  );
}

// ─── Expandable match row ─────────────────────────────────────────────────────

function MatchSection({ item }) {
  const [open, setOpen] = useState(false);
  const pred = item.prediction || {};
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
        <svg
          className={`w-3.5 h-3.5 text-zinc-600 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
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
  const [items, setItems] = useState([]);

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

  const sorted = [...items]
    .filter(i => i.prediction?.xg_home != null)
    .sort((a, b) => (a.match_date || "").localeCompare(b.match_date || ""));

  return (
    <div className="max-w-3xl mx-auto content-pad">
      <div className="px-4 pt-5 pb-3">
        <h1 className="text-base font-semibold text-white">Monte Carlo Sim</h1>
        <p className="text-xs text-zinc-600 mt-0.5">
          {dateLabel} · {N_SIMS.toLocaleString()} Poisson simulations per match · tap to expand
        </p>
      </div>

      {status === "idle" && (
        <div className="px-4 py-12 text-center text-xs text-zinc-600">Loading…</div>
      )}
      {status === "computing" && (
        <div className="px-4 py-3 text-xs text-zinc-600">
          Predictions still computing — showing available data
        </div>
      )}
      {status === "error" && (
        <div className="px-4 py-3 text-xs text-red-500">
          Failed to load. Is the backend running?
        </div>
      )}
      {(status === "ready" || status === "computing") && sorted.length === 0 && (
        <div className="px-4 py-12 text-center text-xs text-zinc-600">
          No matches with xG data available yet
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
