import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import PredictionsPage from "./pages/PredictionsPage";
import MatchesPage from "./pages/MatchesPage";
import StandingsPage from "./pages/StandingsPage";

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<PredictionsPage />} />
          <Route path="/matches" element={<MatchesPage />} />
          <Route path="/standings" element={<StandingsPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
