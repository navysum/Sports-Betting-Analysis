import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
// Root URL for endpoints not under /api prefix (e.g. /accuracy)
const ROOT_URL = BASE_URL.replace(/\/api$/, "");

const api = axios.create({ baseURL: BASE_URL });
const rootApi = axios.create({ baseURL: ROOT_URL });

// ── Competitions ──────────────────────────────────────────────────────────────
export const getCompetitions = () => api.get("/matches/competitions");

// ── Fixtures & Results ────────────────────────────────────────────────────────
export const getTodayMatches = () => api.get("/matches/today");

export const getUpcomingMatches = (competition = "PL", daysAhead = 7) =>
  api.get("/matches/upcoming", { params: { competition, days_ahead: daysAhead } });

export const getResults = (competition = "PL", limit = 30) =>
  api.get("/matches/results", { params: { competition, limit } });

// ── Standings ─────────────────────────────────────────────────────────────────
export const getStandings = (competition = "PL") =>
  api.get("/matches/standings", { params: { competition } });

// ── Team ──────────────────────────────────────────────────────────────────────
export const searchTeam = (q) =>
  api.get("/matches/search", { params: { q } });

export const getTeamForm = (teamId, limit = 10) =>
  api.get(`/matches/team/${teamId}/form`, { params: { limit } });

export const getTeamUpcoming = (teamId, limit = 5) =>
  api.get(`/matches/team/${teamId}/upcoming`, { params: { limit } });

// ── Predictions ───────────────────────────────────────────────────────────────
export const getUpcomingPredictions = (competition = "PL", daysAhead = 1) =>
  api.get("/predictions/upcoming", {
    params: { competition, days_ahead: daysAhead },
  });

export const predictMatch = ({
  homeTeamId,
  awayTeamId,
  competitionCode = "PL",
  homeTeamName = "",
  awayTeamName = "",
  matchDate = "",
}) =>
  api.post("/predictions/predict", {
    home_team_id: homeTeamId,
    away_team_id: awayTeamId,
    competition_code: competitionCode,
    home_team_name: homeTeamName,
    away_team_name: awayTeamName,
    match_date: matchDate,
  });

// ── Today predictions (cached / preloaded) ────────────────────────────────────
export const getTodayPredictions = () => api.get("/predictions/today");
export const triggerPreload      = () => api.post("/predictions/preload");

// ── Stats / Accuracy ──────────────────────────────────────────────────────────
export const getAccuracy = () => rootApi.get("/accuracy");

// ── Admin / Retrain ───────────────────────────────────────────────────────────
export const triggerRetrain    = () => api.post("/admin/retrain");
export const getRetrainStatus  = () => api.get("/admin/retrain/status");
export const getBacktest       = (minEdge = 0.05) =>
  api.get("/admin/backtest", { params: { min_edge: minEdge } });
export const getCLVStats       = (days = 30) =>
  api.get("/admin/clv-stats", { params: { days } });
