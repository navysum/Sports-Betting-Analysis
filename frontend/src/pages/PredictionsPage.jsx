import { useState, useEffect } from "react";
import { getUpcomingPredictions } from "../services/api";
import PredictionCard from "../components/PredictionCard";
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
  const [comp, setComp]   = useState("PL");
  const [days, setDays]   = useState(1);
  const [filter, setFilter] = useState("all");
  const [raw, setRaw]     = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setRaw([]);
    getUpcomingPredictions(comp, days)
      .then(r => setRaw(r.data.predictions || []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [comp, days]);

  const predictions = raw
    .filter(p => {
      const pred = p.prediction || {};
      if (isFlat(pred)) return false;
      if (filter === "strong") return (pred.stars || 0) >= 4;
      if (filter === "value")  return (pred.value_bets || []).length > 0;
      return true;
    })
    .sort((a, b) => (b.prediction?.confidence || 0) - (a.prediction?.confidence || 0));

  const hiddenCount = raw.filter(p => isFlat(p.prediction)).length;

  return (
    <div className="max-w-3xl mx-auto content-pad">
      {/* Header */}
      <div className="px-4 pt-5 pb-4 flex flex-col sm:flex-row sm:items-center gap-3">
        <div className="flex-1">
          <h1 className="text-base font-semibold text-white">Predictions</h1>
          <p className="text-xs text-zinc-600 mt-0.5">ML-powered · Win / Draw / Loss · O2.5 · BTTS</p>
        </div>
        <CompetitionSelector value={comp} onChange={setComp} />
      </div>

      {/* Controls */}
      <div className="px-4 pb-4 flex gap-2 flex-wrap">
        {/* Days */}
        <div className="flex rounded border border-zinc-800 overflow-hidden">
          {DAYS.map(d => (
            <button key={d.value} onClick={() => setDays(d.value)}
              className={`px-3 py-1.5 text-xs transition-colors ${
                days === d.value ? "bg-zinc-800 text-white" : "text-zinc-500 hover:text-zinc-300"
              }`}>
              {d.label}
            </button>
          ))}
        </div>
        {/* Filter */}
        <div className="flex rounded border border-zinc-800 overflow-hidden">
          {FILTERS.map(f => (
            <button key={f.key} onClick={() => setFilter(f.key)}
              className={`px-3 py-1.5 text-xs transition-colors ${
                filter === f.key ? "bg-zinc-800 text-white" : "text-zinc-500 hover:text-zinc-300"
              }`}>
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="px-4 py-16 text-center">
          <p className="text-xs text-zinc-600">Computing predictions…</p>
          <p className="text-xs text-zinc-700 mt-1">Fetching live data + running models — 30–90s</p>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="px-4 text-xs text-red-500">{error}</div>
      )}

      {/* Results */}
      {!loading && !error && (
        <div className="px-4 space-y-2">
          {hiddenCount > 0 && (
            <p className="text-xs text-zinc-700 pb-1">
              {hiddenCount} match{hiddenCount !== 1 ? "es" : ""} hidden (no data)
            </p>
          )}
          {predictions.length === 0 ? (
            <div className="py-12 text-center">
              <p className="text-sm text-zinc-500">No predictions found</p>
              {filter !== "all" && (
                <button onClick={() => setFilter("all")} className="text-xs text-green-500 mt-2">
                  Show all
                </button>
              )}
            </div>
          ) : (
            <>
              <p className="text-xs text-zinc-700 pb-1">
                {predictions.length} prediction{predictions.length !== 1 ? "s" : ""} · by confidence
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {predictions.map((p, i) => (
                  <PredictionCard key={p.api_match_id || i} data={p} />
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
