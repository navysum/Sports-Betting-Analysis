import { useState, useEffect } from "react";
import { getResults } from "../services/api";
import CompetitionSelector from "../components/CompetitionSelector";

function resultBadge(match) {
  const hg = match?.score?.fullTime?.home;
  const ag = match?.score?.fullTime?.away;
  if (hg == null || ag == null) return null;
  if (hg > ag) return { label: "H", cls: "bg-green-700 text-green-100" };
  if (hg === ag) return { label: "D", cls: "bg-yellow-700 text-yellow-100" };
  return { label: "A", cls: "bg-blue-700 text-blue-100" };
}

export default function MatchesPage() {
  const [competition, setCompetition] = useState("PL");
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getResults(competition, 30)
      .then((res) => setMatches(res.data.matches || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [competition]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <h1 className="text-2xl font-bold text-white">Recent Results</h1>
        <CompetitionSelector value={competition} onChange={setCompetition} />
      </div>

      {loading && <p className="text-slate-400 text-center py-12">Loading…</p>}
      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-xl p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      <div className="space-y-2">
        {matches.map((m) => {
          const badge = resultBadge(m);
          const date = new Date(m.utcDate).toLocaleDateString("en-GB", {
            day: "numeric", month: "short",
          });
          const hg = m?.score?.fullTime?.home;
          const ag = m?.score?.fullTime?.away;

          return (
            <div
              key={m.id}
              className="bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 flex items-center gap-4"
            >
              <span className="text-xs text-slate-500 w-14 shrink-0">{date}</span>

              <div className="flex-1 flex items-center justify-end gap-2">
                {m.homeTeam?.crest && (
                  <img src={m.homeTeam.crest} alt="" className="w-5 h-5 object-contain" />
                )}
                <span className="text-sm font-medium">
                  {m.homeTeam?.shortName || m.homeTeam?.name}
                </span>
              </div>

              <div className="flex items-center gap-1 shrink-0">
                <span className="font-mono text-lg font-bold text-white">{hg ?? "–"}</span>
                <span className="text-slate-500">:</span>
                <span className="font-mono text-lg font-bold text-white">{ag ?? "–"}</span>
              </div>

              <div className="flex-1 flex items-center gap-2">
                {m.awayTeam?.crest && (
                  <img src={m.awayTeam.crest} alt="" className="w-5 h-5 object-contain" />
                )}
                <span className="text-sm font-medium">
                  {m.awayTeam?.shortName || m.awayTeam?.name}
                </span>
              </div>

              {badge && (
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-bold w-6 text-center ${badge.cls}`}
                >
                  {badge.label}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
