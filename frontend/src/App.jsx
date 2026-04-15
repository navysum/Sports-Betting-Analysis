import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import MobileNav from "./components/MobileNav";
import HomePage from "./pages/HomePage";
import PredictionsPage from "./pages/PredictionsPage";
import ResultsPage from "./pages/ResultsPage";
import StandingsPage from "./pages/StandingsPage";
import StatsPage from "./pages/StatsPage";
import TeamPage from "./pages/TeamPage";
import DistributionsPage from "./pages/DistributionsPage";
import MonteCarloPage from "./pages/MonteCarloPage";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-dvh flex flex-col">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/"             element={<HomePage />} />
            <Route path="/predictions"  element={<PredictionsPage />} />
            <Route path="/results"      element={<ResultsPage />} />
            <Route path="/standings"    element={<StandingsPage />} />
            <Route path="/stats"          element={<StatsPage />} />
            <Route path="/distributions" element={<DistributionsPage />} />
            <Route path="/montecarlo"    element={<MonteCarloPage />} />
            <Route path="/team/:teamId"  element={<TeamPage />} />
            <Route path="/matches"       element={<ResultsPage />} />
          </Routes>
        </main>
        <MobileNav />
      </div>
    </BrowserRouter>
  );
}
