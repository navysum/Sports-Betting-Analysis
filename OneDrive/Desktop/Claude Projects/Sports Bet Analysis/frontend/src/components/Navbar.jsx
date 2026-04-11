import { Link, useLocation } from "react-router-dom";

const links = [
  { to: "/", label: "Predictions" },
  { to: "/matches", label: "Matches" },
  { to: "/standings", label: "Standings" },
];

export default function Navbar() {
  const { pathname } = useLocation();

  return (
    <nav className="bg-slate-900 border-b border-slate-700 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 flex items-center justify-between h-16">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-2xl">⚽</span>
          <span className="font-bold text-xl text-green-400 tracking-tight">
            SoccerBet<span className="text-white">AI</span>
          </span>
        </Link>

        <div className="flex gap-1">
          {links.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                pathname === l.to
                  ? "bg-green-600 text-white"
                  : "text-slate-300 hover:text-white hover:bg-slate-800"
              }`}
            >
              {l.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
