import { useState, useEffect } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import { getTeamForm, getTeamUpcoming } from "../services/api";

function resultFor(match, teamId) {
  const hg = match?.score?.fullTime?.home;
  const ag = match?.score?.fullTime?.away;
  if (hg == null || ag == null) return null;
  const isHome = match?.homeTeam?.id === teamId;
  if (hg === ag) return "D";
  return isHome ? (hg > ag ? "W" : "L") : (ag > hg ? "W" : "L");
}

const R_STYLE = {
  W: { dot: "bg-green-500", text: "text-green-400" },
  D: { dot: "bg-yellow-500", text: "text-yellow-400" },
  L: { dot: "bg-red-500",   text: "text-red-400" },
};

function fmtDate(s) {
  if (!s) return "—";
  return new Date(s).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}
function fmtKick(s) {
  if (!s) return "TBD";
  return new Date(s).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

export default function TeamPage() {
  const { teamId } = useParams();
  const { state }  = useLocation();
  const navigate   = useNavigate();
  const id         = parseInt(teamId);

  const [team]     = useState(state?.team || null);
  const [form, setForm]         = useState([]);
  const [upcoming, setUpcoming] = useState([]);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    Promise.allSettled([getTeamForm(id, 10), getTeamUpcoming(id, 5)]).then(([f, u]) => {
      if (f.status === "fulfilled") setForm(f.value.data.matches || []);
      if (u.status === "fulfilled") setUpcoming(u.value.data.matches || []);
      setLoading(false);
    });
  }, [id]);

  const teamName = team?.shortName || team?.name || `Team #${id}`;
  const last5    = form.slice(-5).map(m => resultFor(m, id)).filter(Boolean);
  const wins     = form.filter(m => resultFor(m, id) === "W").length;
  const draws    = form.filter(m => resultFor(m, id) === "D").length;
  const losses   = form.filter(m => resultFor(m, id) === "L").length;

  return (
    <div className="max-w-3xl mx-auto content-pad">
      {/* Back */}
      <button onClick={() => navigate(-1)}
        className="px-4 pt-4 pb-0 flex items-center gap-1 text-xs text-zinc-600 hover:text-zinc-300 transition-colors">
        ← Back
      </button>

      {/* Header */}
      <div className="px-4 pt-3 pb-5 flex items-center gap-4">
        {team?.crest && <img src={team.crest} alt="" className="w-12 h-12 object-contain" />}
        <div>
          <h1 className="text-lg font-semibold text-white">{teamName}</h1>
          {team?.area?.name && <p className="text-xs text-zinc-600">{team.area.name}</p>}
          {/* Form dots */}
          {last5.length > 0 && (
            <div className="flex items-center gap-1.5 mt-1.5">
              {last5.map((r, i) => (
                <div key={i} className={`w-2.5 h-2.5 rounded-full ${R_STYLE[r]?.dot || "bg-zinc-700"}`}
                     title={r} />
              ))}
            </div>
          )}
        </div>
      </div>

      {loading && <div className="px-4 py-8 text-xs text-zinc-600 text-center">Loading…</div>}

      {!loading && (
        <>
          {/* Stat pills */}
          {form.length > 0 && (
            <div className="px-4 pb-5 flex gap-2">
              {[
                { label: "P", value: form.length, cls: "text-zinc-300" },
                { label: "W", value: wins,         cls: "text-green-400" },
                { label: "D", value: draws,        cls: "text-yellow-400" },
                { label: "L", value: losses,       cls: "text-red-400" },
              ].map(s => (
                <div key={s.label} className="flex-1 border border-zinc-800 rounded-md py-2 text-center">
                  <p className={`text-lg font-semibold tabular-nums ${s.cls}`}>{s.value}</p>
                  <p className="text-[11px] text-zinc-600">{s.label}</p>
                </div>
              ))}
            </div>
          )}

          {/* Upcoming */}
          {upcoming.length > 0 && (
            <div className="px-4 mb-6">
              <h2 className="text-xs font-medium text-zinc-500 uppercase tracking-wide mb-2">Upcoming</h2>
              <div className="border border-zinc-800 rounded-md overflow-hidden">
                {upcoming.map((m, i) => {
                  const isHome = m.homeTeam?.id === id;
                  const opp    = isHome ? m.awayTeam : m.homeTeam;
                  return (
                    <div key={m.id}
                      className="flex items-center gap-3 px-3 py-2.5 border-b border-zinc-800/50 last:border-0">
                      <span className="text-xs text-zinc-600 shrink-0">{fmtKick(m.utcDate)}</span>
                      <span className="text-xs text-zinc-600 shrink-0 w-8">{isHome ? "H" : "A"}</span>
                      {opp?.crest && <img src={opp.crest} alt="" className="w-4 h-4 object-contain shrink-0 opacity-80" />}
                      <span className="text-sm flex-1">{opp?.shortName || opp?.name}</span>
                      <span className="text-[11px] text-zinc-700">{m.competition?.code}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Form */}
          {form.length > 0 && (
            <div className="px-4">
              <h2 className="text-xs font-medium text-zinc-500 uppercase tracking-wide mb-2">
                Recent Form
              </h2>
              <div className="border border-zinc-800 rounded-md overflow-hidden">
                {[...form].reverse().map((m) => {
                  const r = resultFor(m, id);
                  const rs = R_STYLE[r];
                  const isHome = m.homeTeam?.id === id;
                  const opp = isHome ? m.awayTeam : m.homeTeam;
                  const hg = m?.score?.fullTime?.home ?? "?";
                  const ag = m?.score?.fullTime?.away ?? "?";
                  const myG = isHome ? hg : ag;
                  const oppG = isHome ? ag : hg;

                  return (
                    <div key={m.id}
                      className="flex items-center gap-3 px-3 py-2.5 border-b border-zinc-800/50 last:border-0">
                      {/* Result dot */}
                      <div className={`w-2 h-2 rounded-full shrink-0 ${rs?.dot || "bg-zinc-700"}`} />
                      {/* Date */}
                      <span className="text-xs text-zinc-600 shrink-0 w-16 tabular-nums">{fmtDate(m.utcDate)}</span>
                      {/* H/A */}
                      <span className="text-[11px] text-zinc-700 w-4 shrink-0">{isHome ? "H" : "A"}</span>
                      {/* Opponent */}
                      {opp?.crest && <img src={opp.crest} alt="" className="w-4 h-4 object-contain shrink-0 opacity-80" />}
                      <span className="text-sm flex-1 truncate">{opp?.shortName || opp?.name}</span>
                      {/* Score */}
                      <span className={`text-sm font-mono font-semibold tabular-nums ${rs?.text || "text-zinc-400"}`}>
                        {myG}–{oppG}
                      </span>
                      {/* Comp */}
                      <span className="text-[11px] text-zinc-700 w-8 text-right shrink-0">{m.competition?.code}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {form.length === 0 && (
            <p className="px-4 text-sm text-zinc-600 text-center py-8">No match data available.</p>
          )}
        </>
      )}
    </div>
  );
}
