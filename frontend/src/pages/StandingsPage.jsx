import { useState, useEffect } from "react";
import { getStandings } from "../services/api";
import CompetitionSelector from "../components/CompetitionSelector";

export default function StandingsPage() {
  const [comp, setComp] = useState("PL");
  const [table, setTable] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getStandings(comp)
      .then(r => setTable(r.data.table || []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [comp]);

  function zone(i, total) {
    if (comp === "CL") return null;
    if (i < 4) return "bg-green-500";
    if (i === 4) return "bg-blue-500";
    if (i >= total - 3) return "bg-red-500";
    return null;
  }

  return (
    <div className="max-w-3xl mx-auto content-pad">
      <div className="px-4 pt-5 pb-4 flex items-center justify-between">
        <h1 className="text-base font-semibold text-white">Standings</h1>
        <CompetitionSelector value={comp} onChange={setComp} />
      </div>

      {loading && <div className="px-4 py-12 text-center text-xs text-zinc-600">Loading…</div>}
      {error && <div className="px-4 text-xs text-red-500">{error}</div>}

      {!loading && !error && table.length > 0 && (
        <>
          {/* Column headers */}
          <div className="px-4 pb-1 grid text-[11px] text-zinc-600 font-medium"
               style={{ gridTemplateColumns: "1.5rem 1fr 1.8rem 1.8rem 1.8rem 1.8rem 2.5rem 2.5rem" }}>
            <span>#</span>
            <span>Club</span>
            <span className="text-center">P</span>
            <span className="text-center text-green-600">W</span>
            <span className="text-center text-yellow-600">D</span>
            <span className="text-center text-red-600">L</span>
            <span className="text-center">GD</span>
            <span className="text-center text-white">Pts</span>
          </div>

          <div className="border-t border-zinc-800">
            {table.map((row, i) => {
              const z = zone(i, table.length);
              return (
                <div key={row.team?.id}
                  className="grid items-center px-4 py-2 border-b border-zinc-800/40
                             hover:bg-zinc-900/50 transition-colors"
                  style={{ gridTemplateColumns: "1.5rem 1fr 1.8rem 1.8rem 1.8rem 1.8rem 2.5rem 2.5rem" }}>
                  {/* Pos with zone colour */}
                  <div className="flex items-center gap-1.5">
                    {z && <div className={`w-0.5 h-4 rounded-full ${z}`} />}
                    <span className="text-xs text-zinc-500 tabular-nums">{row.position}</span>
                  </div>
                  {/* Team */}
                  <div className="flex items-center gap-2 min-w-0">
                    {row.team?.crest && (
                      <img src={row.team.crest} alt="" className="w-4 h-4 object-contain shrink-0 opacity-80" />
                    )}
                    <span className="text-sm truncate">{row.team?.shortName || row.team?.name}</span>
                  </div>
                  <span className="text-xs text-zinc-500 text-center tabular-nums">{row.playedGames}</span>
                  <span className="text-xs text-green-600 text-center tabular-nums">{row.won}</span>
                  <span className="text-xs text-yellow-600 text-center tabular-nums">{row.draw}</span>
                  <span className="text-xs text-red-600 text-center tabular-nums">{row.lost}</span>
                  <span className="text-xs text-zinc-400 text-center tabular-nums">
                    {row.goalDifference > 0 ? `+${row.goalDifference}` : row.goalDifference}
                  </span>
                  <span className="text-sm font-semibold text-white text-center tabular-nums">{row.points}</span>
                </div>
              );
            })}
          </div>

          {comp !== "CL" && (
            <div className="px-4 pt-3 flex gap-4 text-[11px] text-zinc-600">
              <span className="flex items-center gap-1.5"><span className="w-1 h-3 rounded-full bg-green-500 inline-block"/>UCL</span>
              <span className="flex items-center gap-1.5"><span className="w-1 h-3 rounded-full bg-blue-500 inline-block"/>UEL</span>
              <span className="flex items-center gap-1.5"><span className="w-1 h-3 rounded-full bg-red-500 inline-block"/>Relegation</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
