function ProbBar({ label, prob, color }) {
  const pct = prob != null ? Math.round(prob * 100) : 0;
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-12 text-slate-400 text-xs">{label}</span>
      <div className="flex-1 bg-slate-700 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-8 text-right font-mono text-slate-200">{pct}%</span>
    </div>
  );
}

function OutcomeBadge({ outcome }) {
  const map = {
    HOME: { label: "Home Win", cls: "bg-green-700 text-green-100" },
    DRAW: { label: "Draw", cls: "bg-yellow-700 text-yellow-100" },
    AWAY: { label: "Away Win", cls: "bg-blue-700 text-blue-100" },
  };
  const { label, cls } = map[outcome] || { label: outcome, cls: "bg-slate-600 text-slate-200" };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${cls}`}>
      {label}
    </span>
  );
}

export default function PredictionCard({ data }) {
  const { home_team, away_team, home_team_crest, away_team_crest, match_date, prediction } = data;

  const date = match_date
    ? new Date(match_date).toLocaleDateString("en-GB", {
        weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
      })
    : "—";

  const confidence = prediction?.confidence != null
    ? Math.round(prediction.confidence * 100)
    : null;

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 hover:border-green-600 transition-colors">
      {/* Teams */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 flex-1">
          {home_team_crest && (
            <img src={home_team_crest} alt="" className="w-7 h-7 object-contain" />
          )}
          <span className="font-semibold text-sm truncate">{home_team}</span>
        </div>

        <div className="px-3 text-slate-500 font-bold text-xs">VS</div>

        <div className="flex items-center gap-2 flex-1 justify-end">
          <span className="font-semibold text-sm truncate text-right">{away_team}</span>
          {away_team_crest && (
            <img src={away_team_crest} alt="" className="w-7 h-7 object-contain" />
          )}
        </div>
      </div>

      {/* Date */}
      <p className="text-xs text-slate-500 mb-3">{date}</p>

      {/* Prediction */}
      {prediction?.predicted_outcome ? (
        <>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-slate-400">Predicted:</span>
            <OutcomeBadge outcome={prediction.predicted_outcome} />
            {confidence && (
              <span className="text-xs text-slate-400 ml-auto">
                {confidence}% confident
              </span>
            )}
          </div>
          <div className="space-y-1.5">
            <ProbBar label="Home" prob={prediction.home_win_prob} color="bg-green-500" />
            <ProbBar label="Draw" prob={prediction.draw_prob} color="bg-yellow-500" />
            <ProbBar label="Away" prob={prediction.away_win_prob} color="bg-blue-500" />
          </div>
        </>
      ) : (
        <p className="text-xs text-slate-500 italic">Prediction unavailable</p>
      )}
    </div>
  );
}
