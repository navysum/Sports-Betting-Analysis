import { useState, useEffect, useCallback } from "react";
import { getBestBets } from "../services/api";

const GRADE_COLORS = {
  A: "text-green-400 border-green-700 bg-green-900/20",
  B: "text-blue-400 border-blue-700 bg-blue-900/20",
  C: "text-yellow-400 border-yellow-700 bg-yellow-900/20",
  D: "text-orange-400 border-orange-700 bg-orange-900/20",
  F: "text-red-400 border-red-700 bg-red-900/20",
};

const REC_COLORS = {
  "STRONG BET": "text-green-400 bg-green-900/30 border-green-700",
  "BET":        "text-emerald-400 bg-emerald-900/30 border-emerald-700",
  "SMALL BET":  "text-blue-400 bg-blue-900/30 border-blue-700",
  "WATCHLIST":  "text-yellow-400 bg-yellow-900/30 border-yellow-700",
  "PASS":       "text-zinc-500 bg-zinc-900/30 border-zinc-700",
  "AVOID":      "text-red-400 bg-red-900/30 border-red-700",
};

const RISK_COLORS = {
  LOW:    "text-green-500",
  MEDIUM: "text-yellow-500",
  HIGH:   "text-red-500",
};

function GradeBadge({ grade }) {
  const cls = GRADE_COLORS[grade] || GRADE_COLORS.F;
  return (
    <span className={`inline-flex items-center justify-center w-7 h-7 rounded border text-sm font-bold ${cls}`}>
      {grade}
    </span>
  );
}

function RecBadge({ rec }) {
  const cls = REC_COLORS[rec] || REC_COLORS.PASS;
  return (
    <span className={`inline-block px-2 py-0.5 rounded border text-[11px] font-medium ${cls}`}>
      {rec}
    </span>
  );
}

function ScoreBar({ score }) {
  const pct = Math.round((score / 10) * 100);
  const color = score >= 8.5 ? "bg-green-500" : score >= 7 ? "bg-blue-500" : score >= 6 ? "bg-yellow-500" : "bg-zinc-600";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-zinc-400 tabular-nums w-8 text-right">{score.toFixed(1)}</span>
    </div>
  );
}

function BetCard({ bet }) {
  const [expanded, setExpanded] = useState(false);
  const ai = bet.ai || {};
  const pred = bet.prediction_summary || {};
  const warnings = ai.warnings || [];
  const reasoning = ai.reasoning || [];

  const kickoff = bet.match_date
    ? new Date(bet.match_date).toLocaleString("en-GB", {
        weekday: "short", day: "numeric", month: "short",
        hour: "2-digit", minute: "2-digit",
      })
    : "—";

  return (
    <div className={`border rounded-lg p-4 space-y-3 ${
      ["STRONG BET", "BET"].includes(ai.recommendation)
        ? "border-green-800/60 bg-zinc-900/80"
        : "border-zinc-800 bg-zinc-900/40"
    }`}>
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            {bet.home_team_crest && (
              <img src={bet.home_team_crest} alt="" className="w-4 h-4 object-contain opacity-80" />
            )}
            <span className="text-sm font-semibold truncate">{bet.home_team}</span>
            <span className="text-zinc-600 text-xs">vs</span>
            <span className="text-sm font-semibold truncate">{bet.away_team}</span>
            {bet.away_team_crest && (
              <img src={bet.away_team_crest} alt="" className="w-4 h-4 object-contain opacity-80" />
            )}
          </div>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            <span className="text-[11px] text-zinc-600">{bet.competition}</span>
            <span className="text-[11px] text-zinc-700">·</span>
            <span className="text-[11px] text-zinc-600 tabular-nums">{kickoff}</span>
          </div>
        </div>
        <GradeBadge grade={ai.grade} />
      </div>

      {/* Market + Recommendation */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-sm font-medium text-white">{ai.market}</span>
        <RecBadge rec={ai.recommendation} />
        <span className={`text-xs font-medium ${RISK_COLORS[ai.risk_level] || "text-zinc-500"}`}>
          {ai.risk_level} RISK
        </span>
      </div>

      {/* Score bar */}
      <ScoreBar score={ai.score || 0} />

      {/* Key metrics row */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-500">
        {pred.best_edge > 0 && (
          <span className="text-green-400">Edge {(pred.best_edge * 100).toFixed(1)}%</span>
        )}
        {pred.confidence && (
          <span>Conf {Math.round(pred.confidence * 100)}%</span>
        )}
        {pred.over25_prob != null && (
          <span className={pred.over25_prob >= 0.5 ? "text-zinc-300" : ""}>
            O2.5 {Math.round(pred.over25_prob * 100)}%
          </span>
        )}
        {pred.xg_home != null && (
          <span>xG {pred.xg_home?.toFixed(2)}–{pred.xg_away?.toFixed(2)}</span>
        )}
        {pred.stars && (
          <span>
            {[1,2,3,4,5].map(i => (
              <span key={i} className={i <= pred.stars ? "text-yellow-400" : "text-zinc-700"}>★</span>
            ))}
          </span>
        )}
      </div>

      {/* Fallback flags */}
      {pred.fallback_flags && Object.values(pred.fallback_flags).some(Boolean) && (
        <div className="flex flex-wrap gap-1">
          {pred.fallback_flags.used_xg_fallback && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-900/30 text-yellow-600 border border-yellow-900">xG proxy</span>
          )}
          {pred.fallback_flags.used_dc_fallback && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-orange-900/30 text-orange-600 border border-orange-900">No DC</span>
          )}
          {pred.fallback_flags.used_global_model && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500 border border-zinc-700">Global model</span>
          )}
          {pred.fallback_flags.used_approx_devig && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500 border border-zinc-700">Approx devig</span>
          )}
        </div>
      )}

      {/* Expandable reasoning */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="text-[11px] text-zinc-600 hover:text-zinc-400 transition-colors"
      >
        {expanded ? "▲ Hide reasoning" : "▼ Show reasoning"}
      </button>

      {expanded && (
        <div className="space-y-2 border-t border-zinc-800 pt-2">
          {reasoning.length > 0 && (
            <div>
              <p className="text-[10px] text-zinc-600 uppercase tracking-wide mb-1">Why</p>
              <ul className="space-y-0.5">
                {reasoning.map((r, i) => (
                  <li key={i} className="text-xs text-zinc-300 flex items-start gap-1.5">
                    <span className="text-green-500 mt-0.5 shrink-0">✓</span>{r}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {warnings.length > 0 && (
            <div>
              <p className="text-[10px] text-zinc-600 uppercase tracking-wide mb-1">Warnings</p>
              <ul className="space-y-0.5">
                {warnings.map((w, i) => (
                  <li key={i} className="text-xs text-yellow-600 flex items-start gap-1.5">
                    <span className="mt-0.5 shrink-0">⚠</span>{w}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {/* Score breakdown */}
          {ai.components && (
            <div>
              <p className="text-[10px] text-zinc-600 uppercase tracking-wide mb-1">Score breakdown</p>
              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-zinc-500">
                {Object.entries(ai.components).map(([k, v]) => (
                  <span key={k} className="flex justify-between">
                    <span>{k.replace("_score", "").replace("_", " ")}</span>
                    <span className="tabular-nums text-zinc-300">{v.toFixed(1)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function BestBetsPage() {
  const [bets, setBets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [minScore, setMinScore] = useState(6.0);
  const [minGrade, setMinGrade] = useState("C");
  const [date, setDate] = useState(null);
  const [total, setTotal] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getBestBets({ minScore, minGrade });
      setBets(res.data.best_bets || []);
      setTotal(res.data.total || 0);
      setDate(res.data.date);
      if (res.data.status === "no_predictions") {
        setError("No predictions loaded yet for today. Check back soon.");
      }
    } catch (e) {
      setError("Failed to load best bets. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, [minScore, minGrade]);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="max-w-3xl mx-auto px-4 py-6 space-y-4 pb-20 md:pb-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-bold">Best Bets</h1>
          {date && <p className="text-xs text-zinc-600 mt-0.5">AI-graded picks for {date}</p>}
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <label className="flex items-center gap-1.5 text-xs text-zinc-500">
            Min score
            <select
              value={minScore}
              onChange={e => setMinScore(parseFloat(e.target.value))}
              className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-1.5 py-1"
            >
              {[5.0, 6.0, 7.0, 7.5, 8.0, 8.5].map(v => (
                <option key={v} value={v}>{v.toFixed(1)}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1.5 text-xs text-zinc-500">
            Min grade
            <select
              value={minGrade}
              onChange={e => setMinGrade(e.target.value)}
              className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-1.5 py-1"
            >
              {["A", "B", "C", "D"].map(g => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </label>
          <button
            onClick={load}
            className="text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-3 py-1 rounded transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* How it works */}
      <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3 text-xs text-zinc-500 space-y-1">
        <p className="font-medium text-zinc-400">How AI grading works</p>
        <p>Each bet is scored 0–10 using: Edge (35%) · Confidence (25%) · Historical ROI (20%) · CLV (10%) · Data quality (10%)</p>
        <div className="flex flex-wrap gap-2 pt-1">
          {[["A", "8.5+", "green"], ["B", "7.0+", "blue"], ["C", "6.0+", "yellow"], ["D", "5.0+", "orange"]].map(([g, t, c]) => (
            <span key={g} className={`text-${c}-400`}>{g} = {t}</span>
          ))}
        </div>
      </div>

      {/* Content */}
      {loading && (
        <div className="text-center py-12 text-zinc-600 text-sm">Analyzing today's matches…</div>
      )}

      {error && (
        <div className="text-center py-8 text-zinc-500 text-sm">{error}</div>
      )}

      {!loading && !error && bets.length === 0 && (
        <div className="text-center py-12 space-y-2">
          <p className="text-zinc-500 text-sm">No bets meet the current criteria.</p>
          <p className="text-zinc-700 text-xs">Try lowering the minimum score or grade.</p>
        </div>
      )}

      {!loading && bets.length > 0 && (
        <>
          <p className="text-xs text-zinc-600">{total} eligible bet{total !== 1 ? "s" : ""} found</p>
          <div className="space-y-3">
            {bets.map((bet, i) => (
              <BetCard key={`${bet.home_team}-${bet.away_team}-${i}`} bet={bet} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
