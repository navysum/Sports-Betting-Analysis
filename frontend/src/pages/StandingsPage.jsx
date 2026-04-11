import { useState, useEffect } from "react";
import { getStandings } from "../services/api";
import CompetitionSelector from "../components/CompetitionSelector";

export default function StandingsPage() {
  const [competition, setCompetition] = useState("PL");
  const [table, setTable] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getStandings(competition)
      .then((res) => setTable(res.data.table || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [competition]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <h1 className="text-2xl font-bold text-white">Standings</h1>
        <CompetitionSelector value={competition} onChange={setCompetition} />
      </div>

      {loading && <p className="text-slate-400 text-center py-12">Loading…</p>}
      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-xl p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700 text-xs uppercase">
                <th className="px-3 py-3 text-left w-8">#</th>
                <th className="px-3 py-3 text-left">Team</th>
                <th className="px-3 py-3 text-center">P</th>
                <th className="px-3 py-3 text-center">W</th>
                <th className="px-3 py-3 text-center">D</th>
                <th className="px-3 py-3 text-center">L</th>
                <th className="px-3 py-3 text-center">GD</th>
                <th className="px-3 py-3 text-center font-bold text-white">Pts</th>
              </tr>
            </thead>
            <tbody>
              {table.map((row, i) => (
                <tr
                  key={row.team?.id}
                  className={`border-b border-slate-700/50 hover:bg-slate-700/40 transition-colors ${
                    i < 4 ? "border-l-2 border-l-green-500" :
                    i === 4 ? "border-l-2 border-l-blue-500" :
                    i >= table.length - 3 ? "border-l-2 border-l-red-500" : ""
                  }`}
                >
                  <td className="px-3 py-2.5 text-slate-400">{row.position}</td>
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      {row.team?.crest && (
                        <img src={row.team.crest} alt="" className="w-5 h-5 object-contain" />
                      )}
                      <span className="font-medium">{row.team?.shortName || row.team?.name}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2.5 text-center text-slate-400">{row.playedGames}</td>
                  <td className="px-3 py-2.5 text-center text-green-400">{row.won}</td>
                  <td className="px-3 py-2.5 text-center text-yellow-400">{row.draw}</td>
                  <td className="px-3 py-2.5 text-center text-red-400">{row.lost}</td>
                  <td className="px-3 py-2.5 text-center text-slate-300">
                    {row.goalDifference > 0 ? `+${row.goalDifference}` : row.goalDifference}
                  </td>
                  <td className="px-3 py-2.5 text-center font-bold text-white">{row.points}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="px-4 py-2 flex gap-4 text-xs text-slate-500 border-t border-slate-700">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 bg-green-500 rounded-sm" /> Champions League
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 bg-blue-500 rounded-sm" /> Europa League
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 bg-red-500 rounded-sm" /> Relegation
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
