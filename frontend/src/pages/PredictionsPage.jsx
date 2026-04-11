import { useState, useEffect } from "react";
import { getUpcomingPredictions } from "../services/api";
import PredictionCard from "../components/PredictionCard";
import CompetitionSelector from "../components/CompetitionSelector";

export default function PredictionsPage() {
  const [competition, setCompetition] = useState("PL");
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getUpcomingPredictions(competition, 7)
      .then((res) => setPredictions(res.data.predictions || []))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [competition]);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Match Predictions</h1>
          <p className="text-slate-400 text-sm mt-1">
            ML-powered Win / Draw / Loss predictions for upcoming fixtures
          </p>
        </div>
        <CompetitionSelector value={competition} onChange={setCompetition} />
      </div>

      {loading && (
        <div className="text-center py-20 text-slate-400">
          <div className="text-4xl mb-3">⚽</div>
          <p>Loading predictions…</p>
        </div>
      )}

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-xl p-4 text-red-300 text-sm">
          {error}. Make sure the backend is running and your API key is set.
        </div>
      )}

      {!loading && !error && predictions.length === 0 && (
        <div className="text-center py-20 text-slate-500">
          No upcoming matches found for this competition in the next 7 days.
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {predictions.map((p) => (
          <PredictionCard key={p.match_id} data={p} />
        ))}
      </div>
    </div>
  );
}
