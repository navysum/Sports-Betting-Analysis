const OC = {
  HOME: { label: "Home", color: "text-green-400" },
  DRAW: { label: "Draw", color: "text-yellow-400" },
  AWAY: { label: "Away", color: "text-blue-400" },
};

const GRADE_COLORS = {
  A: "text-green-400 border-green-700",
  B: "text-blue-400 border-blue-700",
  C: "text-yellow-400 border-yellow-700",
  D: "text-orange-400 border-orange-700",
  F: "text-red-400 border-red-700",
};

const REC_COLORS = {
  "STRONG BET": "text-green-400 bg-green-900/30",
  "BET":        "text-emerald-400 bg-emerald-900/20",
  "SMALL BET":  "text-blue-400 bg-blue-900/20",
  "WATCHLIST":  "text-yellow-500 bg-yellow-900/10",
  "PASS":       "text-zinc-600",
  "AVOID":      "text-red-500",
};

function Bar({ prob, color }) {
  const w = prob != null ? Math.round(prob * 100) : 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${w}%` }} />
      </div>
      <span className="text-zinc-500 tabular-nums w-8 text-right">{w}%</span>
    </div>
  );
}

function Stars({ n = 1 }) {
  return (
    <span className="text-xs">
      {[1,2,3,4,5].map(i => (
        <span key={i} className={i <= n ? "text-yellow-400" : "text-zinc-700"}>★</span>
      ))}
    </span>
  );
}

function AIBadge({ ai }) {
  if (!ai || !ai.grade) return null;
  const cls = GRADE_COLORS[ai.grade] || GRADE_COLORS.F;
  const recCls = REC_COLORS[ai.recommendation] || REC_COLORS.PASS;
  return (
    <div className="flex items-center gap-1.5">
      <span className={`inline-flex items-center justify-center w-5 h-5 rounded border text-[10px] font-bold ${cls}`}>
        {ai.grade}
      </span>
      {ai.recommendation && ai.recommendation !== "PASS" && (
        <span className={`text-[10px] font-medium px-1 rounded ${recCls}`}>
          {ai.recommendation}
        </span>
      )}
    </div>
  );
}

export default function PredictionCard({ data }) {
  const { home_team, away_team, match_date, competition, prediction: pred = {} } = data;

  const kickoff = match_date
    ? new Date(match_date).toLocaleString("en-GB", {
        weekday: "short", day: "numeric", month: "short",
        hour: "2-digit", minute: "2-digit",
      })
    : "—";

  const oc = OC[pred.predicted_outcome];
  const conf = pred.confidence != null ? Math.round(pred.confidence * 100) : null;
  const vbets = pred.value_bets || [];
  const scores = pred.correct_scores || [];
  const isFlat = pred.predicted_outcome && (
    Math.max(pred.home_win_prob, pred.draw_prob, pred.away_win_prob) -
    Math.min(pred.home_win_prob, pred.draw_prob, pred.away_win_prob) < 0.08
  );

  // Extract best AI recommendation from ai_analysis
  const aiAnalysis = pred.ai_analysis;
  const bestAI = aiAnalysis?.best_recommendation;

  // Fallback flags summary
  const flags = pred.fallback_flags || {};
  const hasFallbacks = Object.values(flags).some(Boolean);

  return (
    <div className={`border rounded-md p-3 space-y-2.5 ${
      vbets.length ? "border-green-900/60 bg-zinc-900/60" : "border-zinc-800 bg-zinc-900/40"
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-zinc-600">{competition}</span>
        <span className="text-[11px] text-zinc-600 tabular-nums">{kickoff}</span>
      </div>

      {/* Teams */}
      <div className="flex items-center gap-2">
        {data.home_team_crest && (
          <img src={data.home_team_crest} alt="" className="w-4 h-4 object-contain opacity-80" />
        )}
        <span className="text-sm font-medium flex-1 truncate">{home_team}</span>
        <span className="text-xs text-zinc-700">vs</span>
        <span className="text-sm font-medium flex-1 truncate text-right">{away_team}</span>
        {data.away_team_crest && (
          <img src={data.away_team_crest} alt="" className="w-4 h-4 object-contain opacity-80" />
        )}
      </div>

      {/* No prediction */}
      {!pred.predicted_outcome && (
        <p className="text-xs text-zinc-600 italic">No prediction</p>
      )}

      {pred.predicted_outcome && isFlat && (
        <p className="text-xs text-zinc-600">Insufficient data</p>
      )}

      {pred.predicted_outcome && !isFlat && (
        <>
          {/* Outcome row */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`text-sm font-medium ${oc?.color || "text-zinc-300"}`}>
              {oc?.label}
            </span>
            <Stars n={pred.stars} />
            {conf && <span className="ml-auto text-xs text-zinc-600 tabular-nums">{conf}%</span>}
            {bestAI && <AIBadge ai={bestAI} />}
          </div>

          {/* Bars */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1 text-[11px] text-zinc-600">
              <span className="w-8">H</span><Bar prob={pred.home_win_prob} color="bg-green-500" />
            </div>
            <div className="flex items-center gap-1 text-[11px] text-zinc-600">
              <span className="w-8">D</span><Bar prob={pred.draw_prob} color="bg-yellow-500" />
            </div>
            <div className="flex items-center gap-1 text-[11px] text-zinc-600">
              <span className="w-8">A</span><Bar prob={pred.away_win_prob} color="bg-blue-500" />
            </div>
          </div>

          {/* Metrics */}
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-zinc-600">
            {pred.over25_prob != null && (
              <span className={pred.over25_prob >= 0.5 ? "text-zinc-300" : ""}>
                O2.5 {Math.round(pred.over25_prob * 100)}%
              </span>
            )}
            {pred.btts_prob != null && (
              <span className={pred.btts_prob >= 0.5 ? "text-zinc-300" : ""}>
                BTTS {Math.round(pred.btts_prob * 100)}%
              </span>
            )}
            {pred.xg_home != null && (
              <span>xG {pred.xg_home.toFixed(2)}–{pred.xg_away?.toFixed(2)}</span>
            )}
          </div>

          {/* Correct scores */}
          {scores.length > 0 && (
            <div className="flex gap-3 text-xs text-zinc-600 flex-wrap">
              {scores.slice(0, 4).map(s => (
                <span key={s.score}>
                  <span className="text-zinc-300 font-mono">{s.score}</span>{" "}
                  {Math.round(s.prob * 100)}%
                </span>
              ))}
            </div>
          )}

          {/* Value bets */}
          {vbets.length > 0 && (
            <div className="flex flex-wrap gap-2 text-xs">
              {vbets.map((vb, i) => {
                const label = vb.split("(")[0].trim();
                const nums = vb.match(/(\d+)%/g);
                const edge = nums?.length >= 2 ? parseInt(nums[0]) - parseInt(nums[1]) : null;
                return (
                  <span key={i} className="text-green-500">
                    ↑ {label}{edge != null ? ` +${edge}%` : ""}
                  </span>
                );
              })}
            </div>
          )}

          {/* Fallback quality flags */}
          {hasFallbacks && (
            <div className="flex flex-wrap gap-1">
              {flags.used_xg_fallback && (
                <span className="text-[10px] px-1 py-0.5 rounded bg-yellow-900/20 text-yellow-700 border border-yellow-900/50">xG proxy</span>
              )}
              {flags.used_dc_fallback && (
                <span className="text-[10px] px-1 py-0.5 rounded bg-orange-900/20 text-orange-700 border border-orange-900/50">No DC</span>
              )}
              {flags.used_global_model && (
                <span className="text-[10px] px-1 py-0.5 rounded bg-zinc-800 text-zinc-600 border border-zinc-700">Global mdl</span>
              )}
            </div>
          )}

          {/* AI Score (if eligible) */}
          {bestAI && bestAI.eligible && bestAI.score > 0 && (
            <div className="flex items-center gap-2 text-[11px] text-zinc-600 border-t border-zinc-800/50 pt-2">
              <span>AI score</span>
              <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    bestAI.score >= 8.5 ? "bg-green-500" :
                    bestAI.score >= 7.0 ? "bg-blue-500" :
                    bestAI.score >= 6.0 ? "bg-yellow-500" : "bg-zinc-600"
                  }`}
                  style={{ width: `${Math.round((bestAI.score / 10) * 100)}%` }}
                />
              </div>
              <span className="tabular-nums text-zinc-400">{bestAI.score?.toFixed(1)}</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
