import { useState, useEffect, useRef } from "react";
import { getUpcomingPredictions, getTodayPredictions, triggerPreload } from "../services/api";
import PredictionCard from "../components/PredictionCard";
import { COMPETITIONS } from "../components/CompetitionSelector";
import CompetitionSelector from "../components/CompetitionSelector";

const DAYS = [
  { value: 1, label: "Today" },
  { value: 3, label: "3 days" },
  { value: 7, label: "7 days" },
];
const FILTERS = [
  { key: "all",    label: "All" },
  { key: "strong", label: "Strong" },
  { key: "value",  label: "Value" },
];

function isFlat(pred) {
  if (!pred) return true;
  const { home_win_prob: h = 0, draw_prob: d = 0, away_win_prob: a = 0 } = pred;
  return Math.max(h, d, a) - Math.min(h, d, a) < 0.08;
}

export default function PredictionsPage() {
  const [comp, setComp]   = useState("ALL");
  const [days, setDays]   = useState(1);
  const [filter, setFilter] = useState("all");
  const [raw, setRaw]     = useState([]);
  const [cacheStatus, setCacheStatus] = useState(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  async function fetchTodayCache() {
    const r = await getTodayPredictions();
    const { status, predictions = [], done = 0, total = 0 } = r.data;
    setCacheStatus(status);
    setProgress({ done, total });
    setRaw(predictions);
    return status;
  }

  function startPolling() {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await fetchTodayCache();
        if (s === "ready" || s === "error") clearInterval(pollRef.current);
      } catch {}
    }, 5000);
  }

  // Today-cache path — re-runs only when days changes to/from 1
  useEffect(() => {
    clearInterval(pollRef.current);
    if (days !== 1) return;

    setRaw([]);
    setError(null);
    setCacheStatus(null);
    setLoading(true);

    fetchTodayCache()
      .then((s) => {
        if (s === "idle") {
          triggerPreload().catch(() => {});
          startPolling();
        } else if (s === "computing") {
          startPolling();
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    return () => clearInterval(pollRef.current);
  }, [days]); // eslint-disable-line react-hooks/exhaustive-deps

  // Live upcoming path — re-runs when comp or days (>1) changes
  useEffect(() => {
    if (days === 1) return;

    clearInterval(pollRef.current);
    setRaw([]);
    setError(null);
    setLoading(true);

    const effectiveComp = comp === "ALL" ? "PL" : comp;
    getUpcomingPredictions(effectiveComp, days)
      .then((r) => setRaw(r.data.predictions || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    return () => clearInterval(pollRef.current);
  }, [days, comp]);

  const predictions = raw
    .filter((p) => {
      const pred = p.prediction || {};
      if (isFlat(pred)) return false;
      if (days === 1 && comp !== "ALL" && p.competition_code !== comp) return false;
      if (filter === "strong") return (pred.stars || 0) >= 4;
      if (filter === "value")  return (pred.value_bets || []).length > 0;
      return true;
    })
    .sort((a, b) => (b.prediction?.confidence || 0) - (a.prediction?.confidence || 0));

  const hiddenCount = raw.filter((p) => {
    if (days === 1 && comp !== "ALL" && p.competition_code !== comp) return false;
    return isFlat(p.prediction);
  }).length;

  const isComputing = days === 1 && cacheStatus === "computing";

  return (
    <div className="max-w-3xl mx-auto content-pad">
      {/* Header */}
      <div className="px-4 pt-5 pb-4 flex flex-col sm:flex-row sm:items-center gap-3">
        <div className="flex-1">
          <h1 className="text-base font-semibold text-white">Predictions</h1>
          <p className="text-xs text-zinc-600 mt-0.5">ML-powered · Win / Draw / Loss · O2.5 · BTTS</p>
        </div>
        {days === 1 ? (
          <select
            value={comp}
            onChange={(e) => setComp(e.target.value)}
            className="bg-zinc-900 border border-zinc-700 text-zinc-200 rounded px-2.5 py-1.5
                       text-sm outline-none focus:border-zinc-500 cursor-pointer"
          >
            <option value="ALL">All competitions</option>
            {Object.entries(COMPETITIONS).map(([code, name]) => (
              <option key={code} value={code}>{name}</option>
            ))}
          </select>
        ) : (
          <CompetitionSelector value={comp === "ALL" ? "PL" : comp} onChange={setComp} />
        )}
      </div>

      {/* Controls */}
      <div className="px-4 pb-4 flex gap-2 flex-wrap">
        <div className="flex rounded border border-zinc-800 overflow-hidden">
          {DAYS.map((d) => (
            <button key={d.value} onClick={() => setDays(d.value)}
              className={`px-3 py-1.5 text-xs transition-colors ${
                days === d.value ? "bg-zinc-800 text-white" : "text-zinc-500 hover:text-zinc-300"
              }`}>
              {d.label}
            </button>
          ))}
        </div>
        <div className="flex rounded border border-zinc-800 overflow-hidden">
          {FILTERS.map((f) => (
            <button key={f.key} onClick={() => setFilter(f.key)}
              className={`px-3 py-1.5 text-xs transition-colors ${
                filter === f.key ? "bg-zinc-800 text-white" : "text-zinc-500 hover:text-zinc-300"
              }`}>
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Initial load spinner */}
      {loading && (
        <div className="px-4 py-16 text-center">
          <p className="text-xs text-zinc-600">
            {days === 1 ? "Loading…" : "Computing predictions… (30–90s)"}
          </p>
        </div>
      )}

      {/* Computing progress — shown while cache is still building, even with partial results */}
      {!loading && isComputing && predictions.length === 0 && (
        <div className="px-4 py-16 text-center space-y-1.5">
          <p className="text-xs text-zinc-500">Computing today's predictions…</p>
          {progress.total > 0 && (
            <p className="text-xs text-zinc-700">{progress.done} / {progress.total} matches done</p>
          )}
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="px-4 text-xs text-red-500">{error}</div>
      )}

      {/* Predictions list */}
      {!loading && !error && predictions.length > 0 && (
        <div className="px-4 space-y-2">
          {hiddenCount > 0 && (
            <p className="text-xs text-zinc-700 pb-1">
              {hiddenCount} match{hiddenCount !== 1 ? "es" : ""} hidden (no data)
            </p>
          )}
          <p className="text-xs text-zinc-700 pb-1">
            {predictions.length} prediction{predictions.length !== 1 ? "s" : ""} · by confidence
            {isComputing && " · more coming…"}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {predictions.map((p, i) => (
              <PredictionCard key={p.api_match_id || i} data={p} />
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && !isComputing && predictions.length === 0 && (
        <div className="px-4 py-12 text-center">
          <p className="text-sm text-zinc-500">No predictions found</p>
          {filter !== "all" && (
            <button onClick={() => setFilter("all")}
              className="text-xs text-green-500 mt-2 block mx-auto hover:text-green-400 transition-colors">
              Show all
            </button>
          )}
          {days === 1 && comp !== "ALL" && (
            <button onClick={() => setComp("ALL")}
              className="text-xs text-green-500 mt-2 block mx-auto hover:text-green-400 transition-colors">
              All competitions
            </button>
          )}
        </div>
      )}
    </div>
  );
}
