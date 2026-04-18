import { useState, useEffect, useRef } from "react";
import { getAccuracy, triggerRetrain, getRetrainStatus, getBacktest } from "../services/api";

function pct(v) { return v != null ? `${Math.round(v * 100)}%` : "—"; }

function StatRow({ label, value, sub }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-zinc-800/50 last:border-0">
      <div>
        <span className="text-sm text-zinc-300">{label}</span>
        {sub && <span className="text-xs text-zinc-600 ml-2">{sub}</span>}
      </div>
      <span className="text-sm font-mono text-white">{value}</span>
    </div>
  );
}

function WindowSection({ label, stats }) {
  if (!stats) return null;
  const total = stats.total || 0;
  const accPct = stats.result_accuracy != null ? Math.round(stats.result_accuracy * 100) : 0;

  return (
    <div className="mb-6">
      <div className="flex items-baseline gap-2 mb-3">
        <h2 className="text-sm font-medium text-white">{label}</h2>
        <span className="text-xs text-zinc-600">{total} predictions</span>
      </div>

      {total === 0 ? (
        <p className="text-xs text-zinc-600 italic py-2">No settled predictions yet</p>
      ) : (
        <div className="border border-zinc-800 rounded-md overflow-hidden">
          <div className="px-4 py-3 border-b border-zinc-800 flex items-center gap-4">
            <div className="flex-1">
              <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div className="h-full bg-green-500 rounded-full transition-all"
                     style={{ width: `${accPct}%` }} />
              </div>
            </div>
            <span className="text-sm font-mono font-semibold text-green-400 w-12 text-right">
              {pct(stats.result_accuracy)}
            </span>
            <span className="text-xs text-zinc-600">result accuracy</span>
          </div>
          <div className="px-4">
            <StatRow label="Over 2.5" value={pct(stats.over25_accuracy)} />
            <StatRow label="BTTS" value={pct(stats.btts_accuracy)} />
            <StatRow
              label="Log-loss"
              value={stats.log_loss != null ? stats.log_loss.toFixed(3) : "—"}
              sub="lower = better · random ≈ 1.10"
            />
            <StatRow
              label="Brier score"
              value={stats.brier_score != null ? stats.brier_score.toFixed(3) : "—"}
              sub="lower = better · random ≈ 0.67"
            />
          </div>
        </div>
      )}
    </div>
  );
}

const STATUS_LABEL = {
  idle:    "No retrain running",
  running: "Training in progress…",
  done:    "Training complete",
  failed:  "Training failed",
};

const STATUS_COLOR = {
  idle:    "text-zinc-500",
  running: "text-yellow-400",
  done:    "text-green-400",
  failed:  "text-red-400",
};

function RetrainPanel() {
  const [retrainState, setRetrainState] = useState(null);
  const [triggering, setTriggering] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const pollRef = useRef(null);

  async function loadStatus() {
    try {
      const res = await getRetrainStatus();
      setRetrainState(res.data);
      return res.data.status;
    } catch {
      return null;
    }
  }

  function startPolling() {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      const s = await loadStatus();
      if (s === "done" || s === "failed" || s == null) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }, 4000);
  }

  useEffect(() => {
    loadStatus().then(s => {
      if (s === "running") startPolling();
    });
    return () => clearInterval(pollRef.current);
  }, []);

  async function handleRetrain() {
    setTriggering(true);
    try {
      await triggerRetrain();
      await loadStatus();
      startPolling();
    } catch (e) {
      console.error(e);
    } finally {
      setTriggering(false);
    }
  }

  const s = retrainState?.status || "idle";
  const isRunning = s === "running";
  const log = retrainState?.log || [];
  const summary = retrainState?.summary || retrainState?.last_training;

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-white">Model Retraining</h2>
        <button
          onClick={handleRetrain}
          disabled={isRunning || triggering}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors
            ${isRunning || triggering
              ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-500 text-white"
            }`}
        >
          {isRunning ? (
            <>
              <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Training…
            </>
          ) : (
            <>
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round"
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Retrain Model
            </>
          )}
        </button>
      </div>

      <div className="border border-zinc-800 rounded-md overflow-hidden">
        {/* Status row */}
        <div className="px-4 py-3 flex items-center gap-2 border-b border-zinc-800/60">
          {isRunning && (
            <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse shrink-0" />
          )}
          <span className={`text-xs ${STATUS_COLOR[s]}`}>
            {STATUS_LABEL[s] || s}
          </span>
          {retrainState?.started_at && (
            <span className="text-xs text-zinc-700 ml-auto">
              {new Date(retrainState.started_at).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}
        </div>

        {/* Last training summary */}
        {summary && (
          <div className="px-4">
            {summary.samples && (
              <StatRow label="Samples" value={summary.samples.toLocaleString()} />
            )}
            {summary.result_model?.accuracy_mean != null && (
              <StatRow
                label="Result model CV"
                value={pct(summary.result_model.accuracy_mean)}
                sub={`±${pct(summary.result_model.accuracy_std)}`}
              />
            )}
            {summary.goals_model?.accuracy_mean != null && (
              <StatRow label="O/U 2.5 CV" value={pct(summary.goals_model.accuracy_mean)} />
            )}
            {summary.btts_model?.accuracy_mean != null && (
              <StatRow label="BTTS CV" value={pct(summary.btts_model.accuracy_mean)} />
            )}
            {summary.trained_at && (
              <StatRow
                label="Last trained"
                value={new Date(summary.trained_at).toLocaleDateString("en-GB", {
                  day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
                })}
              />
            )}
          </div>
        )}

        {/* Error */}
        {s === "failed" && retrainState?.error && (
          <div className="px-4 py-2">
            <p className="text-xs text-red-400 font-mono break-all">{retrainState.error}</p>
          </div>
        )}

        {/* Log toggle */}
        {log.length > 0 && (
          <div className="border-t border-zinc-800/60">
            <button
              onClick={() => setShowLog(v => !v)}
              className="w-full px-4 py-2 text-left text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
            >
              {showLog ? "Hide" : "Show"} log ({log.length} lines)
            </button>
            {showLog && (
              <div className="px-4 pb-3 max-h-48 overflow-y-auto space-y-0.5">
                {log.map((line, i) => (
                  <p key={i} className="text-[10px] font-mono text-zinc-500 leading-relaxed">{line}</p>
                ))}
              </div>
            )}
          </div>
        )}

        {/* No summary yet */}
        {!summary && s === "idle" && (
          <div className="px-4 py-3">
            <p className="text-xs text-zinc-600">
              No trained model found. Press <span className="text-zinc-400">Retrain Model</span> to train from scratch.
              This fetches historical data and takes ~10–30 minutes.
            </p>
          </div>
        )}
      </div>

      <p className="text-xs text-zinc-700 mt-2">
        Retraining downloads CSV data + fetches this season's API data, then trains 3 XGBoost models.
        Predictions update automatically when done.
      </p>
    </div>
  );
}

function roi(v) {
  if (v == null) return "—";
  const s = v >= 0 ? "+" : "";
  return `${s}${v.toFixed(1)}%`;
}

function BacktestPanel() {
  const [bt, setBt] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    getBacktest()
      .then(r => setBt(r.data))
      .catch(e => setError(e.response?.data?.detail || e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="mb-8">
      <div className="flex items-baseline gap-2 mb-3">
        <h2 className="text-sm font-medium text-white">Backtest (Historical)</h2>
        {bt && (
          <span className="text-xs text-zinc-600">
            {bt.total_matches?.toLocaleString()} matches · last {bt.holdout_pct}%
          </span>
        )}
      </div>

      {loading && <p className="text-xs text-zinc-600 py-4 text-center">Running backtest…</p>}
      {error && <p className="text-xs text-red-500">{error}</p>}

      {bt && !bt.error && (
        <div className="space-y-3">
          {/* Brier calibration */}
          <div className="border border-zinc-800 rounded-md overflow-hidden">
            <div className="px-4 py-2 border-b border-zinc-800/60">
              <span className="text-xs font-medium text-zinc-400">Calibration (Brier score, lower = better)</span>
            </div>
            <div className="px-4">
              <StatRow label="Result" value={bt.brier?.result?.toFixed(4) ?? "—"} sub="random ≈ 0.67" />
              <StatRow label="Over 2.5" value={bt.brier?.over25?.toFixed(4) ?? "—"} sub="random ≈ 0.25" />
              <StatRow label="BTTS" value={bt.brier?.btts?.toFixed(4) ?? "—"} sub="random ≈ 0.25" />
              <StatRow label="Over 3.5" value={bt.brier?.over35?.toFixed(4) ?? "—"} sub="random ≈ 0.20" />
            </div>
          </div>

          {/* Result market staking */}
          <div className="border border-zinc-800 rounded-md overflow-hidden">
            <div className="px-4 py-2 border-b border-zinc-800/60">
              <span className="text-xs font-medium text-zinc-400">Result market staking (£1/bet)</span>
            </div>
            <div className="px-4">
              <StatRow
                label="Flat — every match"
                value={<span className={bt.flat?.roi >= 0 ? "text-green-400" : "text-red-400"}>{roi(bt.flat?.roi)}</span>}
                sub={`${bt.flat?.bets?.toLocaleString() ?? "—"} bets`}
              />
              <StatRow
                label={`Value — edge ≥ ${bt.value?.min_edge_pct ?? 5}%`}
                value={<span className={bt.value?.roi >= 0 ? "text-green-400" : "text-red-400"}>{roi(bt.value?.roi)}</span>}
                sub={`${bt.value?.bets?.toLocaleString() ?? "—"} bets · ${bt.value?.win_rate?.toFixed(1) ?? "—"}% WR`}
              />
              <StatRow
                label="Kelly — fractional (0.25×)"
                value={<span className={bt.kelly?.pnl >= 0 ? "text-green-400" : "text-red-400"}>
                  {bt.kelly?.final_bankroll != null ? `£${bt.kelly.final_bankroll.toFixed(0)}` : "—"}
                </span>}
                sub={`from £${bt.kelly?.starting_bankroll ?? 100} · max DD ${bt.kelly?.max_drawdown_pct?.toFixed(1) ?? "—"}%`}
              />
            </div>
          </div>

          {/* O/U 2.5 staking */}
          {bt.over25_flat && (
            <div className="border border-zinc-800 rounded-md overflow-hidden">
              <div className="px-4 py-2 border-b border-zinc-800/60">
                <span className="text-xs font-medium text-zinc-400">Over/Under 2.5 staking (£1/bet)</span>
              </div>
              <div className="px-4">
                <StatRow
                  label="Flat — every match"
                  value={<span className={bt.over25_flat?.roi >= 0 ? "text-green-400" : "text-red-400"}>{roi(bt.over25_flat?.roi)}</span>}
                  sub={`${bt.over25_flat?.bets?.toLocaleString() ?? "—"} bets`}
                />
                <StatRow
                  label={`Value — edge ≥ ${bt.over25_value?.min_edge_pct ?? 5}%`}
                  value={<span className={bt.over25_value?.roi >= 0 ? "text-green-400" : "text-red-400"}>{roi(bt.over25_value?.roi)}</span>}
                  sub={`${bt.over25_value?.bets?.toLocaleString() ?? "—"} bets · ${bt.over25_value?.win_rate?.toFixed(1) ?? "—"}% WR`}
                />
              </div>
            </div>
          )}

          {bt._cached_at && (
            <p className="text-[10px] text-zinc-700 text-right">
              Cached {new Date(bt._cached_at).toLocaleTimeString("en-GB")}
            </p>
          )}
        </div>
      )}

      {bt?.error && (
        <p className="text-xs text-red-400 font-mono">{bt.error}</p>
      )}
    </div>
  );
}

export default function StatsPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getAccuracy()
      .then(r => setData(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-3xl mx-auto content-pad">
      <div className="px-4 pt-5 pb-4">
        <h1 className="text-base font-semibold text-white">Model Stats</h1>
        <p className="text-xs text-zinc-600 mt-0.5">Accuracy against real results</p>
      </div>

      <div className="px-4">
        <RetrainPanel />
        <BacktestPanel />

        {loading && <p className="text-xs text-zinc-600 py-8 text-center">Loading…</p>}
        {error && <p className="text-xs text-red-500">{error}</p>}
        {data && (
          <>
            <WindowSection label="Last 7 days"  stats={data["7d"]} />
            <WindowSection label="Last 30 days" stats={data["30d"]} />
            <WindowSection label="All time"     stats={data["all"]} />
            <p className="text-xs text-zinc-700 mt-2">
              Stats only count predictions saved to the ledger.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
