import { useState, useEffect } from "react";
import { getResults } from "../services/api";
import CompetitionSelector from "../components/CompetitionSelector";

export default function ResultsPage() {
  const [comp, setComp] = useState("PL");
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getResults(comp, 30)
      .then(r => setMatches([...(r.data.matches || [])].reverse()))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [comp]);

  return (
    <div className="max-w-3xl mx-auto content-pad">
      <div className="px-4 pt-5 pb-4 flex items-center justify-between">
        <h1 className="text-base font-semibold text-white">Results</h1>
        <CompetitionSelector value={comp} onChange={setComp} />
      </div>

      {loading && <div className="px-4 py-12 text-center text-xs text-zinc-600">Loading…</div>}
      {error && <div className="px-4 text-xs text-red-500">{error}</div>}

      {!loading && !error && (
        <div className="bg-zinc-900/40 border-t border-zinc-800">
          {matches.map((m) => {
            const hg = m?.score?.fullTime?.home;
            const ag = m?.score?.fullTime?.away;
            const home = m.homeTeam?.shortName || m.homeTeam?.name || "?";
            const away = m.awayTeam?.shortName || m.awayTeam?.name || "?";
            const date = m.utcDate
              ? new Date(m.utcDate).toLocaleDateString("en-GB", { day: "numeric", month: "short" })
              : "—";

            let accentColor = "bg-zinc-700";
            if (hg != null && ag != null) {
              if (hg > ag) accentColor = "bg-green-500";
              else if (hg === ag) accentColor = "bg-yellow-500";
              else accentColor = "bg-blue-500";
            }

            return (
              <div key={m.id}
                className="flex items-center gap-3 px-4 py-2.5 border-b border-zinc-800/50 last:border-0">
                {/* Date */}
                <span className="text-xs text-zinc-600 w-14 shrink-0 tabular-nums">{date}</span>

                {/* Result dot */}
                <div className={`w-1 h-6 rounded-full shrink-0 ${accentColor} opacity-70`} />

                {/* Home */}
                <div className="flex items-center gap-1.5 flex-1 min-w-0 justify-end">
                  <span className="text-sm truncate text-right">{home}</span>
                  {m.homeTeam?.crest && (
                    <img src={m.homeTeam.crest} alt="" className="w-4 h-4 object-contain shrink-0 opacity-80" />
                  )}
                </div>

                {/* Score */}
                <div className="font-mono font-semibold text-white text-sm tabular-nums shrink-0 w-12 text-center">
                  {hg ?? "–"} : {ag ?? "–"}
                </div>

                {/* Away */}
                <div className="flex items-center gap-1.5 flex-1 min-w-0">
                  {m.awayTeam?.crest && (
                    <img src={m.awayTeam.crest} alt="" className="w-4 h-4 object-contain shrink-0 opacity-80" />
                  )}
                  <span className="text-sm truncate">{away}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
