import { useState, useEffect } from "react";
import { kickoffTime } from "../utils/time";
import { getTodayPredictions } from "../services/api";

// Poisson probability: P(k; λ) = e^(-λ) * λ^k / k!
function poisson(lambda, k) {
  if (lambda <= 0) return k === 0 ? 1 : 0;
  let logP = -lambda + k * Math.log(lambda);
  for (let i = 1; i <= k; i++) logP -= Math.log(i);
  return Math.exp(logP);
}

function goalDist(lambda, maxGoal = 4) {
  const probs = [];
  let cumulative = 0;
  for (let k = 0; k < maxGoal; k++) {
    const p = poisson(lambda, k);
    probs.push({ k, p });
    cumulative += p;
  }
  probs.push({ k: maxGoal, p: Math.max(0, 1 - cumulative), label: `${maxGoal}+` });
  return probs;
}

// SVG bar chart for goal distributions
function GoalChart({ lambda, color, label }) {
  const dist = goalDist(lambda || 1.2);
  const maxP = Math.max(...dist.map(d => d.p), 0.01);
  const barW = 28;
  const gap = 8;
  const chartH = 72;
  const totalW = dist.length * (barW + gap) - gap;

  return (
    <div>
      <p className="text-[10px] text-zinc-500 mb-1">{label}</p>
      <svg width={totalW} height={chartH + 20} className="overflow-visible">
        {dist.map((d, i) => {
          const barH = Math.max(2, (d.p / maxP) * chartH);
          const x = i * (barW + gap);
          const y = chartH - barH;
          return (
            <g key={i}>
              <rect x={x} y={y} width={barW} height={barH}
                    fill={color} rx={2} opacity={0.85} />
              <text x={x + barW / 2} y={chartH + 12} textAnchor="middle"
                    fontSize={9} fill="#71717a">
                {d.label ?? d.k}
              </text>
              <text x={x + barW / 2} y={y - 3} textAnchor="middle"
                    fontSize={8} fill="#a1a1aa">
                {Math.round(d.p * 100)}%
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// Horizontal probability bar
function HBar({ label, pct, color, highlight }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`w-12 shrink-0 ${highlight ? "text-zinc-200" : "text-zinc-500"}`}>{label}</span>
      <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-zinc-500 tabular-nums w-8 text-right">{pct}%</span>
    </div>
  );
}

// Score matrix from correct_scores[]
function ScoreMatrix({ scores }) {
  if (!scores?.length) return null;
  const top = scores.slice(0, 9);
  const maxP = Math.max(...top.map(s => s.prob), 0.01);

  return (
    <div className="flex flex-wrap gap-2">
      {top.map((s) => {
        const intensity = s.prob / maxP;
        const opacity = 0.15 + intensity * 0.75;
        return (
          <div key={s.score}
               className="flex flex-col items-center justify-center rounded border border-zinc-700 px-2.5 py-1.5"
               style={{ backgroundColor: `rgba(34,197,94,${opacity * 0.25})`,
                        borderColor: `rgba(34,197,94,${opacity * 0.5})` }}>
            <span className="font-mono text-sm text-zinc-200">{s.score}</span>
            <span className="text-[10px] text-zinc-500">{Math.round(s.prob * 100)}%</span>
          </div>
        );
      })}
    </div>
  );
}

// Over/Under bars — compute from xg if needed, or use over25_prob
function OUBars({ over25Prob, xgHome, xgAway }) {
  // Compute O1.5, O2.5, O3.5 from Poisson joint if xg available
  function ouProb(threshold) {
    if (!xgHome || !xgAway) return null;
    let p = 0;
    for (let h = 0; h <= 7; h++) {
      for (let a = 0; a <= 7; a++) {
        if (h + a > threshold) {
          p += poisson(xgHome, h) * poisson(xgAway, a);
        }
      }
    }
    return p;
  }

  const o15 = ouProb(1);
  const o25 = ouProb(2) ?? over25Prob;
  const o35 = ouProb(3);

  const bars = [
    { label: "O 1.5", p: o15 },
    { label: "O 2.5", p: o25 },
    { label: "O 3.5", p: o35 },
    { label: "U 2.5", p: o25 != null ? 1 - o25 : null },
  ].filter(b => b.p != null);

  if (!bars.length) return null;

  return (
    <div className="space-y-1.5">
      {bars.map(b => (
        <HBar key={b.label} label={b.label} pct={Math.round(b.p * 100)}
              color={b.label.startsWith("O") ? "bg-blue-500" : "bg-orange-500"}
              highlight={b.p >= 0.5} />
      ))}
    </div>
  );
}

function MatchSection({ item }) {
  const [open, setOpen] = useState(false);
  const pred = item.prediction || {};
  const hasPred = pred.predicted_outcome && pred.xg_home != null;

  return (
    <div className="border-b border-zinc-800/60 last:border-0">
      {/* Header row */}
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
        {hasPred ? (
          <span className="text-[10px] text-zinc-600 shrink-0 w-14 text-right">
            xG {pred.xg_home?.toFixed(1)}–{pred.xg_away?.toFixed(1)}
          </span>
        ) : (
          <span className="w-14 shrink-0" />
        )}
        <svg className={`w-3.5 h-3.5 text-zinc-600 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
             fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded distributions */}
      {open && (
        <div className="px-4 pb-5 space-y-5 border-t border-zinc-800/40 pt-4">
          {!hasPred && (
            <p className="text-xs text-zinc-600">No distribution data for this match.</p>
          )}

          {hasPred && (
            <>
              {/* Goal distributions */}
              <div>
                <p className="text-[11px] font-medium text-zinc-400 mb-3 uppercase tracking-wider">
                  Goal Distributions
                </p>
                <div className="flex flex-wrap gap-6 overflow-x-auto pb-1">
                  <GoalChart lambda={pred.xg_home} color="#22c55e" label={`${item.home_team} Goals`} />
                  <GoalChart lambda={pred.xg_away} color="#3b82f6" label={`${item.away_team} Goals`} />
                </div>
              </div>

              {/* Win probabilities */}
              <div>
                <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
                  Result Probability
                </p>
                <div className="space-y-1.5">
                  <HBar label={item.home_team.split(" ").pop()}
                        pct={Math.round((pred.home_win_prob || 0) * 100)}
                        color="bg-green-500"
                        highlight={pred.predicted_outcome === "HOME"} />
                  <HBar label="Draw"
                        pct={Math.round((pred.draw_prob || 0) * 100)}
                        color="bg-yellow-500"
                        highlight={pred.predicted_outcome === "DRAW"} />
                  <HBar label={item.away_team.split(" ").pop()}
                        pct={Math.round((pred.away_win_prob || 0) * 100)}
                        color="bg-blue-500"
                        highlight={pred.predicted_outcome === "AWAY"} />
                </div>
              </div>

              {/* Over/Under */}
              <div>
                <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
                  Goals Market
                </p>
                <OUBars
                  over25Prob={pred.over25_prob}
                  xgHome={pred.xg_home}
                  xgAway={pred.xg_away}
                />
                {pred.btts_prob != null && (
                  <div className="mt-1.5">
                    <HBar label="BTTS"
                          pct={Math.round(pred.btts_prob * 100)}
                          color="bg-purple-500"
                          highlight={pred.btts_prob >= 0.5} />
                  </div>
                )}
              </div>

              {/* Score matrix */}
              {pred.correct_scores?.length > 0 && (
                <div>
                  <p className="text-[11px] font-medium text-zinc-400 mb-2 uppercase tracking-wider">
                    Likely Scores
                  </p>
                  <ScoreMatrix scores={pred.correct_scores} />
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function DistributionsPage() {
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
        <h1 className="text-base font-semibold text-white">Distributions</h1>
        <p className="text-xs text-zinc-600 mt-0.5">{dateLabel} · tap a match to expand</p>
      </div>

      {status === "idle" && (
        <div className="px-4 py-12 text-center text-xs text-zinc-600">Loading…</div>
      )}
      {status === "computing" && (
        <div className="px-4 py-3 text-xs text-zinc-600">Predictions still computing — showing available data</div>
      )}
      {status === "error" && (
        <div className="px-4 py-3 text-xs text-red-500">Failed to load. Is the backend running?</div>
      )}
      {(status === "ready" || status === "computing") && sorted.length === 0 && (
        <div className="px-4 py-12 text-center text-xs text-zinc-600">
          No distribution data available yet
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
