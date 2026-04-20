import { useState, useEffect } from "react";
import { getCLVStats, getEvaluation, getAIPerformance } from "../services/api";

function StatCard({ label, value, sub, positive }) {
  const color = positive === true ? "text-green-400" : positive === false ? "text-red-400" : "text-white";
  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
      <p className="text-[11px] text-zinc-600 uppercase tracking-wide">{label}</p>
      <p className={`text-xl font-bold mt-1 ${color}`}>{value ?? "—"}</p>
      {sub && <p className="text-[11px] text-zinc-600 mt-0.5">{sub}</p>}
    </div>
  );
}

function MarketRow({ market, stats }) {
  const clv = stats.avg_clv;
  const isPos = clv != null && clv > 0;
  return (
    <div className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0 text-sm">
      <span className="text-zinc-300 capitalize">{market.replace("over25", "Over 2.5").replace("over35", "Over 3.5").replace("btts", "BTTS")}</span>
      <div className="flex items-center gap-4 text-xs text-zinc-500">
        <span>{stats.count} predictions</span>
        <span className={isPos ? "text-green-400" : "text-red-400"}>
          {clv != null ? `CLV ${(clv * 100).toFixed(2)}%` : "No CLV"}
        </span>
        <span>{stats.positive_rate != null ? `${Math.round(stats.positive_rate * 100)}% beat close` : "—"}</span>
      </div>
    </div>
  );
}

function CalibrationTable({ rows }) {
  if (!rows || rows.length === 0) return (
    <p className="text-xs text-zinc-600 py-4 text-center">No calibration data yet — needs settled predictions.</p>
  );
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-zinc-600 border-b border-zinc-800">
            <th className="text-left py-1.5 font-normal">Prob bucket</th>
            <th className="text-right py-1.5 font-normal">Count</th>
            <th className="text-right py-1.5 font-normal">Expected</th>
            <th className="text-right py-1.5 font-normal">Actual</th>
            <th className="text-right py-1.5 font-normal">Diff</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.bucket} className="border-b border-zinc-800/50">
              <td className="py-1.5 text-zinc-400">{row.bucket}</td>
              <td className="text-right py-1.5 text-zinc-500">{row.count}</td>
              <td className="text-right py-1.5 text-zinc-400">{(row.expected * 100).toFixed(1)}%</td>
              <td className="text-right py-1.5 text-zinc-300">{(row.actual * 100).toFixed(1)}%</td>
              <td className={`text-right py-1.5 font-medium ${row.diff >= 0 ? "text-green-400" : "text-red-400"}`}>
                {row.diff >= 0 ? "+" : ""}{(row.diff * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RoiTable({ data, label }) {
  if (!data || Object.keys(data).length === 0) return (
    <p className="text-xs text-zinc-600 py-3 text-center">No data yet.</p>
  );
  const entries = Object.entries(data)
    .filter(([, v]) => v.count > 0)
    .sort((a, b) => (b[1].roi ?? -999) - (a[1].roi ?? -999));
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-zinc-600 border-b border-zinc-800">
            <th className="text-left py-1.5 font-normal">{label}</th>
            <th className="text-right py-1.5 font-normal">Bets</th>
            <th className="text-right py-1.5 font-normal">Hit%</th>
            <th className="text-right py-1.5 font-normal">ROI%</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, v]) => (
            <tr key={key} className="border-b border-zinc-800/50">
              <td className="py-1.5 text-zinc-400">{key}</td>
              <td className="text-right py-1.5 text-zinc-500">{v.count}</td>
              <td className="text-right py-1.5 text-zinc-400">
                {v.hit_rate != null ? `${(v.hit_rate * 100).toFixed(0)}%` : "—"}
              </td>
              <td className={`text-right py-1.5 font-medium ${
                v.roi == null ? "text-zinc-600" : v.roi >= 0 ? "text-green-400" : "text-red-400"
              }`}>
                {v.roi != null ? `${v.roi >= 0 ? "+" : ""}${v.roi.toFixed(1)}%` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const TABS = ["CLV", "Calibration", "ROI Report", "AI Performance"];

export default function CLVPage() {
  const [tab, setTab] = useState("CLV");
  const [clvDays, setClvDays] = useState(30);
  const [clvData, setClvData] = useState(null);
  const [evalData, setEvalData] = useState(null);
  const [aiData, setAiData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [clv, ev, ai] = await Promise.all([
          getCLVStats(clvDays),
          getEvaluation(365),
          getAIPerformance(30),
        ]);
        setClvData(clv.data);
        setEvalData(ev.data);
        setAiData(ai.data);
      } catch (e) {
        setError("Failed to load data.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [clvDays]);

  const avgClv = clvData?.avg_clv;
  const posRate = clvData?.positive_clv_rate;

  return (
    <div className="max-w-3xl mx-auto px-4 py-6 space-y-4 pb-20 md:pb-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold">Analytics</h1>
        <p className="text-xs text-zinc-600 mt-0.5">CLV · Calibration · ROI report · AI performance</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-zinc-800">
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-2 text-xs font-medium transition-colors border-b-2 -mb-px ${
              tab === t ? "border-green-500 text-white" : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {loading && <div className="text-center py-10 text-zinc-600 text-sm">Loading…</div>}
      {error && <div className="text-center py-6 text-zinc-500 text-sm">{error}</div>}

      {/* CLV Tab */}
      {!loading && tab === "CLV" && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <label className="text-xs text-zinc-500 flex items-center gap-1.5">
              Lookback
              <select
                value={clvDays}
                onChange={e => setClvDays(parseInt(e.target.value))}
                className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-1.5 py-1"
              >
                {[7, 14, 30, 60, 90, 180, 365].map(d => (
                  <option key={d} value={d}>{d}d</option>
                ))}
              </select>
            </label>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <StatCard
              label="Avg CLV"
              value={avgClv != null ? `${(avgClv * 100).toFixed(2)}%` : "—"}
              sub="positive = consistent edge"
              positive={avgClv != null ? avgClv > 0 : undefined}
            />
            <StatCard
              label="Beat-close rate"
              value={posRate != null ? `${Math.round(posRate * 100)}%` : "—"}
              sub="% beating closing line"
              positive={posRate != null ? posRate >= 0.5 : undefined}
            />
            <StatCard
              label="Total predictions"
              value={clvData?.total_predictions ?? "—"}
              sub="all time"
            />
            <StatCard
              label="With CLV data"
              value={clvData?.predictions_with_clv ?? "—"}
              sub={`last ${clvDays} days`}
            />
          </div>

          {/* CLV by market */}
          {clvData?.by_market && Object.keys(clvData.by_market).length > 0 ? (
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
              <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-2">CLV by market</h3>
              {Object.entries(clvData.by_market).map(([m, s]) => (
                <MarketRow key={m} market={m} stats={s} />
              ))}
            </div>
          ) : !loading && (
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-4 text-center">
              <p className="text-sm text-zinc-500">No CLV data yet.</p>
              <p className="text-xs text-zinc-600 mt-1">CLV is computed when Pinnacle closing odds are recorded after kick-off.</p>
            </div>
          )}

          <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3 text-xs text-zinc-600 space-y-1">
            <p className="text-zinc-400 font-medium">What is CLV?</p>
            <p>Closing Line Value (CLV) measures if your model consistently finds odds better than Pinnacle's closing price. Positive avg CLV over many predictions is the strongest indicator of a real betting edge.</p>
          </div>
        </div>
      )}

      {/* Calibration Tab */}
      {!loading && tab === "Calibration" && (
        <div className="space-y-4">
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
            <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-3">Expected vs actual hit rate</h3>
            <CalibrationTable rows={evalData?.calibration || []} />
          </div>
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3 text-xs text-zinc-600">
            <p className="text-zinc-400 font-medium mb-1">Reading the calibration table</p>
            <p>A well-calibrated model shows Diff ≈ 0 across all buckets. Positive diff = model is conservative (actual hit rate higher than predicted). Negative = overconfident.</p>
          </div>
        </div>
      )}

      {/* ROI Report Tab */}
      {!loading && tab === "ROI Report" && evalData && (
        <div className="space-y-4">
          {/* Summary */}
          {evalData.summary && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <StatCard
                label="Total bets"
                value={evalData.summary.count ?? "—"}
                sub={`${evalData.summary.predictions_with_outcome ?? 0} settled`}
              />
              <StatCard
                label="Hit rate"
                value={evalData.summary.hit_rate != null ? `${(evalData.summary.hit_rate * 100).toFixed(0)}%` : "—"}
                positive={evalData.summary.hit_rate != null ? evalData.summary.hit_rate >= 0.5 : undefined}
              />
              <StatCard
                label="ROI"
                value={evalData.summary.roi != null ? `${evalData.summary.roi >= 0 ? "+" : ""}${evalData.summary.roi?.toFixed(1)}%` : "—"}
                positive={evalData.summary.roi != null ? evalData.summary.roi >= 0 : undefined}
              />
              <StatCard
                label="Max drawdown"
                value={evalData.summary.max_drawdown != null ? evalData.summary.max_drawdown.toFixed(2) : "—"}
                sub="units"
              />
            </div>
          )}

          <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
            <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-2">ROI by market</h3>
            <RoiTable data={evalData.by_market} label="Market" />
          </div>
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
            <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-2">ROI by league</h3>
            <RoiTable data={evalData.by_league} label="League" />
          </div>
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
            <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-2">ROI by odds band</h3>
            <RoiTable data={evalData.by_odds_bucket} label="Odds band" />
          </div>
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
            <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-2">ROI by edge bucket</h3>
            <RoiTable data={evalData.by_edge_bucket} label="Edge" />
          </div>
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
            <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-2">ROI by confidence bucket</h3>
            <RoiTable data={evalData.by_confidence_bucket} label="Model prob" />
          </div>
        </div>
      )}

      {/* AI Performance Tab */}
      {!loading && tab === "AI Performance" && aiData && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            <StatCard label="Total decisions" value={aiData.total_decisions ?? "—"} />
            <StatCard label="Settled" value={aiData.settled ?? "—"} />
          </div>

          {aiData.by_grade && Object.keys(aiData.by_grade).length > 0 && (
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
              <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-2">Performance by grade</h3>
              <RoiTable
                data={Object.fromEntries(
                  Object.entries(aiData.by_grade).map(([g, v]) => [
                    `Grade ${g}`,
                    { count: v.count, hit_rate: v.win_rate, roi: v.roi },
                  ])
                )}
                label="Grade"
              />
            </div>
          )}

          {aiData.by_recommendation && Object.keys(aiData.by_recommendation).length > 0 && (
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
              <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-2">Performance by recommendation</h3>
              <RoiTable
                data={Object.fromEntries(
                  Object.entries(aiData.by_recommendation).map(([r, v]) => [
                    r,
                    { count: v.count, hit_rate: v.win_rate, roi: v.roi },
                  ])
                )}
                label="Recommendation"
              />
            </div>
          )}

          {(!aiData.settled || aiData.settled === 0) && (
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-4 text-center">
              <p className="text-sm text-zinc-500">No settled decisions yet.</p>
              <p className="text-xs text-zinc-600 mt-1">Outcomes are recorded after matches settle and results are logged.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
