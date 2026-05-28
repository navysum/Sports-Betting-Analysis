import { useState, useEffect, useRef } from "react";
import { getUpcomingPredictions, getTodayPredictions, triggerPreload } from "../services/api";
import PredictionCard from "../components/PredictionCard";
import { COMPETITIONS } from "../components/CompetitionSelector";
import CompetitionSelector from "../components/CompetitionSelector";

const DAYS = [
  { value: 1,  label: "Today" },
  { value: 3,  label: "3 days" },
  { value: 7,  label: "7 days" },
  { value: 14, label: "14 days" },
  { value: 21, label: "21 days" },
  { value: 30, label: "30 days" },
];

const FILTERS = [
  { key: "all",    label: "All" },
  { key: "strong", label: "Strong" },
  { key: "value",  label: "Value" },
];

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}
function addDays(iso, n) {
  const d = new Date(iso);
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
}
function dateToDaysAhead(iso) {
  const diff = (new Date(iso) - new Date(todayISO())) / 86_400_000;
  return Math.max(1, Math.round(diff));
}

function isFlat(pred) {
  if (!pred) return true;
  const { home_win_prob: h = 0, draw_prob: d = 0, away_win_prob: a = 0 } = pred;
  return Math.max(h, d, a) - Math.min(h, d, a) < 0.08;
}

// Default: 14 days ahead so WC (starts ~June 11) is in range
const DEFAULT_DATE = addDays(todayISO(), 14);

export default function PredictionsPage() {
  const [comp, setComp]             = useState("WC");
  const [days, setDays]             = useState(14);
  const [selectedDate, setSelectedDate] = useState(DEFAULT_DATE); // "" means Today mode
  const [filter, setFilter]         = useState("all");
  const [raw, setRaw]               = useState([]);
  const [cacheStatus, setCacheStatus] = useState(null);
  const [progress, setProgress]     = useState({ done: 0, total: 0 });
  const [loading, setLoading]       = useState(false);
  const [fetching, setFetching]     = useState(false);
  const [bgRefreshing, setBgRefreshing] = useState(false);
  const [cachedAt, setCachedAt]     = useState(null);
  const [error, setError]           = useState(null);
  const pollRef = useRef(null);

  const isToday = selectedDate === ""; // Today mode uses the preload cache

  // ── Today path helpers ────────────────────────────────────────────────────

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

  // ── Upcoming path helpers ─────────────────────────────────────────────────

  async function loadUpcoming(competition, daysAhead, { silent = false, force = false } = {}) {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const r = await getUpcomingPredictions(competition, daysAhead, force);
      const data = r.data;
      setRaw(data.predictions || []);
      if (data.cached_at) setCachedAt(data.cached_at);
      if (data.refreshing) {
        setBgRefreshing(true);
        clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
          try {
            const r2 = await getUpcomingPredictions(competition, daysAhead);
            const d2 = r2.data;
            setRaw(d2.predictions || []);
            if (d2.cached_at) setCachedAt(d2.cached_at);
            if (!d2.refreshing) {
              setBgRefreshing(false);
              clearInterval(pollRef.current);
            }
          } catch {}
        }, 8000);
      } else {
        setBgRefreshing(false);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      if (!silent) setLoading(false);
    }
  }

  // ── Effects ───────────────────────────────────────────────────────────────

  useEffect(() => {
    clearInterval(pollRef.current);
    setBgRefreshing(false);
    if (!isToday) return;

    setRaw([]);
    setError(null);
    setCacheStatus(null);
    setLoading(true);

    fetchTodayCache()
      .then((s) => {
        if (s === "idle") { triggerPreload().catch(() => {}); startPolling(); }
        else if (s === "computing") startPolling();
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    return () => clearInterval(pollRef.current);
  }, [isToday]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (isToday) return;
    clearInterval(pollRef.current);
    const effectiveComp = comp === "ALL" ? "WC" : comp;
    loadUpcoming(effectiveComp, days, { silent: false, force: false });
    return () => clearInterval(pollRef.current);
  }, [days, comp, isToday]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Manual fetch ──────────────────────────────────────────────────────────

  async function handleManualFetch() {
    if (fetching) return;
    setFetching(true);
    clearInterval(pollRef.current);
    setError(null);
    try {
      if (isToday) {
        await triggerPreload();
        setCacheStatus("computing");
        setRaw([]);
        startPolling();
      } else {
        const effectiveComp = comp === "ALL" ? "WC" : comp;
        await loadUpcoming(effectiveComp, days, { silent: true, force: true });
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setFetching(false);
    }
  }

  // ── Date picker handler ───────────────────────────────────────────────────

  function handleDateChange(e) {
    const val = e.target.value;
    setSelectedDate(val);
    if (val) {
      const d = dateToDaysAhead(val);
      setDays(d);
    }
  }

  // ── Derived ───────────────────────────────────────────────────────────────

  const isComputing = isToday && cacheStatus === "computing";
  const isBusy      = fetching || isComputing;

  const predictions = raw
    .filter((p) => {
      const pred = p.prediction || {};
      if (isFlat(pred)) return false;
      if (isToday && comp !== "ALL" && p.competition_code !== comp) return false;
      if (filter === "strong") return (pred.stars || 0) >= 4;
      if (filter === "value")  return (pred.value_bets || []).length > 0;
      return true;
    })
    .sort((a, b) => (b.prediction?.confidence || 0) - (a.prediction?.confidence || 0));

  const hiddenCount = raw.filter((p) => {
    if (isToday && comp !== "ALL" && p.competition_code !== comp) return false;
    return isFlat(p.prediction);
  }).length;

  // Friendly label for the date window
  const windowLabel = isToday
    ? "Today"
    : `Up to ${new Date(selectedDate).toLocaleDateString("en-GB", { day: "numeric", month: "short" })}`;

  return (
    <div className="max-w-3xl mx-auto content-pad">

      {/* ── Header ── */}
      <div className="px-4 pt-5 pb-4 flex flex-col sm:flex-row sm:items-center gap-3">
        <div className="flex-1">
          <h1 className="text-base font-semibold text-white">Predictions</h1>
          <p className="text-xs text-zinc-600 mt-0.5">ML-powered · Win / Draw / Loss · O2.5 · BTTS</p>
        </div>

        {isToday ? (
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
          <CompetitionSelector value={comp === "ALL" ? "WC" : comp} onChange={setComp} />
        )}
      </div>

      {/* ── Controls row ── */}
      <div className="px-4 pb-4 flex gap-2 flex-wrap items-center">

        {/* Day tabs */}
        <div className="flex rounded border border-zinc-800 overflow-hidden">
          {DAYS.map((d) => (
            <button key={d.value} onClick={() => {
              setDays(d.value);
              if (d.value === 1) {
                setSelectedDate("");
              } else {
                setSelectedDate(addDays(todayISO(), d.value));
              }
            }}
              className={`px-3 py-1.5 text-xs transition-colors ${
                days === d.value ? "bg-zinc-800 text-white" : "text-zinc-500 hover:text-zinc-300"
              }`}>
              {d.label}
            </button>
          ))}
        </div>

        {/* Date picker — synced with tabs; picking a custom date deselects tabs */}
        <input
          type="date"
          value={selectedDate}
          min={addDays(todayISO(), 1)}
          max={addDays(todayISO(), 30)}
          onChange={handleDateChange}
          className="bg-zinc-900 border border-zinc-700 text-zinc-300 rounded px-2.5 py-1.5
                     text-xs outline-none focus:border-zinc-500 cursor-pointer [color-scheme:dark]"
        />

        {/* Filter tabs */}
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

        {/* Last updated + manual fetch */}
        <div className="ml-auto flex items-center gap-2">
          {!isToday && cachedAt && !fetching && (
            <span className="text-xs text-zinc-600">
              Updated {new Date(cachedAt).toLocaleString("en-GB", {
                day: "numeric", month: "short",
                hour: "2-digit", minute: "2-digit",
              })}
            </span>
          )}
          <button
            onClick={handleManualFetch}
            disabled={isBusy || loading}
            title="Re-fetch predictions from the API"
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded border
                       border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500
                       transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <span className={fetching ? "inline-block animate-spin" : ""} style={{ display: "inline-block" }}>↻</span>
            {fetching ? "Fetching…" : "Fetch"}
          </button>
        </div>
      </div>

      {/* ── Background refresh indicator ── */}
      {bgRefreshing && !loading && (
        <div className="px-4 pb-2 flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
          <span className="text-xs text-zinc-500">Updating in background…</span>
        </div>
      )}

      {/* ── Initial load spinner ── */}
      {loading && (
        <div className="px-4 py-16 text-center">
          <p className="text-xs text-zinc-600">
            {isToday ? "Loading…" : `Computing predictions for ${windowLabel}… (30–90s)`}
          </p>
        </div>
      )}

      {/* ── Today computing progress ── */}
      {!loading && isComputing && predictions.length === 0 && (
        <div className="px-4 py-16 text-center space-y-1.5">
          <p className="text-xs text-zinc-500">Computing today's predictions…</p>
          {progress.total > 0 && (
            <p className="text-xs text-zinc-700">{progress.done} / {progress.total} matches done</p>
          )}
        </div>
      )}

      {/* ── Error ── */}
      {error && !loading && (
        <div className="px-4 text-xs text-red-500">{error}</div>
      )}

      {/* ── Predictions list ── */}
      {!loading && !error && predictions.length > 0 && (
        <div className="px-4 space-y-2">
          {hiddenCount > 0 && (
            <p className="text-xs text-zinc-700 pb-1">
              {hiddenCount} match{hiddenCount !== 1 ? "es" : ""} hidden (no data)
            </p>
          )}
          <p className="text-xs text-zinc-700 pb-1">
            {predictions.length} prediction{predictions.length !== 1 ? "s" : ""} · {windowLabel} · by confidence
            {isComputing && " · more coming…"}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {predictions.map((p, i) => (
              <PredictionCard key={p.api_match_id || i} data={p} />
            ))}
          </div>
        </div>
      )}

      {/* ── Empty state ── */}
      {!loading && !error && !isComputing && !bgRefreshing && predictions.length === 0 && (
        <div className="px-4 py-12 text-center">
          <p className="text-sm text-zinc-500">No predictions found</p>
          <p className="text-xs text-zinc-600 mt-1">{windowLabel}</p>
          {filter !== "all" && (
            <button onClick={() => setFilter("all")}
              className="text-xs text-green-500 mt-2 block mx-auto hover:text-green-400 transition-colors">
              Show all
            </button>
          )}
          {isToday && comp !== "ALL" && (
            <button onClick={() => setComp("ALL")}
              className="text-xs text-green-500 mt-2 block mx-auto hover:text-green-400 transition-colors">
              All competitions
            </button>
          )}
          <button
            onClick={handleManualFetch}
            disabled={isBusy}
            className="text-xs text-zinc-600 mt-3 block mx-auto hover:text-zinc-400 transition-colors
                       disabled:opacity-40 disabled:cursor-not-allowed"
          >
            ↻ Retry fetch
          </button>
        </div>
      )}

    </div>
  );
}
