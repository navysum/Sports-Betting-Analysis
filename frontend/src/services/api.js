import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

const api = axios.create({ baseURL: BASE_URL });

export const getCompetitions = () => api.get("/matches/competitions");

export const getUpcomingMatches = (competition = "PL", daysAhead = 7) =>
  api.get("/matches/upcoming", { params: { competition, days_ahead: daysAhead } });

export const getResults = (competition = "PL", limit = 20) =>
  api.get("/matches/results", { params: { competition, limit } });

export const getStandings = (competition = "PL") =>
  api.get("/matches/standings", { params: { competition } });

export const getUpcomingPredictions = (competition = "PL", daysAhead = 7) =>
  api.get("/predictions/upcoming", { params: { competition, days_ahead: daysAhead } });

export const predictMatch = (homeTeamId, awayTeamId, competitionCode = "PL") =>
  api.post("/predictions/predict", {
    home_team_id: homeTeamId,
    away_team_id: awayTeamId,
    competition_code: competitionCode,
  });
