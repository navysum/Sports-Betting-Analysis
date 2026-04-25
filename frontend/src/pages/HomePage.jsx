import { useState, useEffect, useRef } from "react";
import { getTodayPredictions, triggerPreload } from "../services/api";

function kickoffTime(utcStr) {
  if (!utcStr) return "—";
  return new Date(utcStr).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

const OC_COLOR = { HOME: "text-green-400", DRAW: "text-yellow-400", AWAY: "text-blue-400" };
const OC_LABEL = { HOME: "Home", DRAW: "Draw", AWAY: "Away" };

function Stars({ n = 1 }) {
  return (
    <span className="text-xs">
      {[1,2,3,4,5].map(i => (
        <span key={i} className={i <= n ? "text-yellow-400" : "text-zinc-700"}>★</span>
      ))}
    </span>
  );
}

function ProbRow({ label, prob, color }) {
  const w = prob != null ? Math.round(prob * 100) : 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-zinc-600 w-8">{label}</span>
      <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${w}%` }} />
      </div>
      <span className="text-zinc-500 tabular-nums w-8 text-right">{w}%</span>
    </div>
  );
}

function MatchRow({ item }) {
  const [open, setOpen] = useState(false);
  const pred = item.prediction || {};
  const oc   = pred.predicted_outcome;
  const vbets = pred.value_bets || [];
  const isFlat = oc && (
    Math.max(pred.home_win_prob, pred.draw_prob, pred.away_win_prob) -
    Math.min(pred.home_win_prob, pred.draw_prob, pred.away_win_prob) < 0.08
  );

  return (
    <div className="border-b border-zinc-800/60 last:border-0">
      {/* Row */}
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-zinc-900/40 transition-colors"
      >
        {/* Time */}
        <span className="text-xs text-zinc-600 tabular-nums w-10 shrink-0">
          {kickoffTime(item.match_date)}
        </span>

        {/* Competition badge */}
        <span className="text-[10px] text-zinc-600 w-20 shrink-0 truncate hidden sm:block">
          {item.competition}
        </span>

        {/* Home */}
        <div className="flex items-center gap-1.5 flex-1 min-w-0">
          {item.home_team_crest && (
            <img src={item.home_team_crest} alt="" className="w-4 h-4 object-contain shrink-0 opacity-80" />
          )}
          <span className="text-sm truncate">{item.home_team}</span>
        </div>

        <span className="text-xs text-zinc-700 shrink-0">—</span>

        {/* Away */}
        <div className="flex items-center gap-1.5 flex-1 min-w-0 justify-end">
          <span className="text-sm truncate text-right">{item.away_team}</span>
          {item.away_team_crest && (
            <img src={item.away_team_crest} alt="" className="w-4 h-4 object-contain shrink-0 opacity-80" />
          )}
        </div>

        {/* Outcome pill (if predicted) */}
        {oc && !isFlat ? (
          <span className={`text-xs font-medium shrink-0 w-10 text-right ${OC_COLOR[oc]}`}>
            {OC_LABEL[oc]}
          </span>
        ) : (
          <span className="w-10 shrink-0" />
        )}

        {/* Expand chevron */}
        <svg className={`w-3.5 h-3.5 text-zinc-600 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
             fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded prediction */}
      {open && (
        <div className="px-4 pb-4 space-y-2.5 border-t border-zinc-800/40">
          {/* Comp on mobile */}
          <p className="text-[11px] text-zinc-600 pt-2 sm:hidden">{item.competition}</p>

          {!oc && <p className="text-xs text-zinc-600 pt-2">No prediction available</p>}
          {oc && isFlat && <p className="text-xs text-zinc-600 pt-2">Insufficient data for this match</p>}

          {oc && !isFlat && (
            <>
              {/* Outcome + stars + conf */}
              <div className="flex items-center gap-3 pt-2">
                <span className={`text-sm font-medium ${OC_COLOR[oc]}`}>{OC_LABEL[oc]} Win</span>
                <Stars n={pred.stars} />
                {pred.confidence && (
                  <span className="ml-auto text-xs text-zinc-600 tabular-nums">
                    {Math.round(pred.confidence * 100)}%
                  </span>
                )}
              </div>

              {/* Prob bars */}
              <div className="space-y-1.5">
                <ProbRow label="Home" prob={pred.home_win_prob} color="bg-green-500" />
                <ProbRow label="Draw" prob={pred.draw_prob}     color="bg-yellow-500" />
                <ProbRow label="Away" prob={pred.away_win_prob} color="bg-blue-500" />
              </div>

              {/* Metrics */}
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-600">
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
                  <span>xG {pred.xg_home.toFixed(2)} – {pred.xg_away?.toFixed(2)}</span>
                )}
              </div>

              {/* Top scores */}
              {pred.correct_scores?.length > 0 && (
                <div className="flex gap-3 flex-wrap text-xs text-zinc-600">
                  {pred.correct_scores.slice(0, 4).map(s => (
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

              {/* Injuries & suspensions */}
              {((pred.home_injuries?.length > 0) || (pred.away_injuries?.length > 0)) && (
                <div className="space-y-0.5">
                  {pred.home_injuries?.length > 0 && (
                    <p className="text-xs text-red-400/80">
                      <span className="text-zinc-500">{item.home_team}:</span>{" "}
                      {pred.home_injuries.map(inj =>
                        `${inj.player} (${inj.type})`
                      ).join(", ")}
                    </p>
                  )}
                  {pred.away_injuries?.length > 0 && (
                    <p className="text-xs text-red-400/80">
                      <span className="text-zinc-500">{item.away_team}:</span>{" "}
                      {pred.away_injuries.map(inj =>
                        `${inj.player} (${inj.type})`
                      ).join(", ")}
                    </p>
                  )}
                </div>
              )}

              {/* Model adjustments applied */}
              {pred.adjustments?.length > 0 && (
                <div className="space-y-0.5">
                  {pred.adjustments.map((adj, i) => (
                    <p key={i} className="text-[11px] text-zinc-600 italic">{adj}</p>
                  ))}
                </div>
              )}

              {/* FBref quality signal */}
              {(pred.home_fbref?.xgd != null || pred.away_fbref?.xgd != null) && (
                <div className="flex gap-4 text-xs text-zinc-600">
                  {pred.home_fbref?.xgd != null && (
                    <span>
                      {item.home_team} xGD{" "}
                      <span className={pred.home_fbref.xgd > 0 ? "text-green-400" : "text-red-400"}>
                        {pred.home_fbref.xgd > 0 ? "+" : ""}{pred.home_fbref.xgd.toFixed(2)}
                      </span>
                    </span>
                  )}
                  {pred.away_fbref?.xgd != null && (
                    <span>
                      {item.away_team} xGD{" "}
                      <span className={pred.away_fbref.xgd > 0 ? "text-green-400" : "text-red-400"}>
                        {pred.away_fbref.xgd > 0 ? "+" : ""}{pred.away_fbref.xgd.toFixed(2)}
                      </span>
                    </span>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function ProgressBar({ done, total, phase }) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const label = phase === "predicting" && total > 0
    ? `Computing predictions… ${done}/${total}`
    : "Fetching today's fixtures…";
  return (
    <div className="px-4 py-3 border-b border-zinc-800">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs text-zinc-500">{label}</span>
        {total > 0 && <span className="text-xs text-zinc-600 tabular-nums">{done}/{total}</span>}
      </div>
      <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
        <div className="h-full bg-green-600 rounded-full transition-all duration-500"
             style={{ width: phase === "predicting" ? `${pct}%` : "100%" }}
             className={phase === "predicting" ? "" : "animate-pulse"} />
      </div>
    </div>
  );
}

export default function HomePage() {
  const [status, setStatus]   = useState("idle");
  const [phase, setPhase]     = useState("starting");
  const [items, setItems]     = useState([]);
  const [done, setDone]       = useState(0);
  const [total, setTotal]     = useState(0);
  const [apiError, setApiError] = useState(null);
  const pollRef               = useRef(null);

  const dateLabel = new Date().toLocaleDateString("en-GB", {
    weekday: "long", day: "numeric", month: "long",
  });

  async function fetchCache() {
    try {
      const res = await getTodayPredictions();
      const d = res.data;
      setStatus(d.status);
      setPhase(d.phase || "starting");
      setItems(d.predictions || []);
      setDone(d.done || 0);
      setTotal(d.total || 0);
      setApiError(d.error || null);
      return d.status;
    } catch {
      setStatus("error");
      setApiError("Cannot reach the backend. Is it running?");
      return "error";
    }
  }

  function startPolling() {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      const s = await fetchCache();
      if (s === "ready" || s === "error") clearInterval(pollRef.current);
    }, 6000);
  }

  async function startPreload() {
    try { await triggerPreload(); } catch {}
    fetchCache();
    startPolling();
  }

  useEffect(() => {
    // Check cache state on mount; also re-trigger if previous run found 0 matches
    fetchCache().then((s) => {
      if (s === "idle") startPreload();
    });

    startPolling();
    return () => clearInterval(pollRef.current);
  }, []);

  // Sort by kickoff time
  const sorted = [...items].sort((a, b) =>
    (a.match_date || "").localeCompare(b.match_date || "")
  );

  return (
    <div className="max-w-3xl mx-auto content-pad">
      {/* Header */}
      <div className="px-4 pt-5 pb-3">
        <h1 className="text-base font-semibold text-white">Today</h1>
        <p className="text-xs text-zinc-600 mt-0.5">{dateLabel}</p>
      </div>

      {/* Progress bar while computing */}
      {status === "computing" && <ProgressBar done={done} total={total} phase={phase} />}

      {/* Error */}
      {status === "error" && (
        <div className="px-4 py-6 space-y-2">
          <p className="text-sm text-red-400">Failed to load matches</p>
          {apiError && (
            <p className="text-xs text-red-400/70 font-mono break-words">{apiError}</p>
          )}
          {apiError?.includes("FOOTBALL_DATA_API_KEY") && (
            <p className="text-xs text-zinc-500 mt-2">
              Set <span className="font-mono text-zinc-300">FOOTBALL_DATA_API_KEY</span> in{" "}
              <span className="font-mono text-zinc-300">backend/.env</span>.
              Get a free key at{" "}
              <span className="font-mono text-zinc-400">football-data.org</span>.
            </p>
          )}
          <button
            onClick={startPreload}
            className="mt-2 text-xs text-green-500 hover:text-green-400 transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* Idle / no matches */}
      {status === "idle" && (
        <div className="px-4 py-12 text-center text-xs text-zinc-600">Starting up…</div>
      )}

      {/* No matches today */}
      {status === "ready" && sorted.length === 0 && (
        <div className="px-4 py-12 text-center">
          <p className="text-sm text-zinc-500">No matches today</p>
          <p className="text-xs text-zinc-600 mt-1">
            None of the tracked leagues (PL, Championship, La Liga, Serie A, Bundesliga, Ligue 1…) have fixtures scheduled today.
          </p>
          <button
            onClick={startPreload}
            className="mt-3 text-xs text-green-500 hover:text-green-400 transition-colors"
          >
            Refresh
          </button>
        </div>
      )}

      {/* Match list */}
      {sorted.length > 0 && (
        <div className="mt-1">
          {status === "computing" && (
            <div className="px-4 py-2 text-xs text-zinc-600">
              {sorted.length} match{sorted.length !== 1 ? "es" : ""} ready so far · more loading…
            </div>
          )}
          {status === "ready" && (
            <div className="px-4 py-2 text-xs text-zinc-600">
              {sorted.length} match{sorted.length !== 1 ? "es" : ""} · tap a row to expand
            </div>
          )}
          <div className="bg-zinc-900/30">
            {sorted.map(item => (
              <MatchRow key={item.api_match_id} item={item} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
