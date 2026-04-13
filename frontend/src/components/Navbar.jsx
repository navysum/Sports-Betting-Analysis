import { useState, useRef, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { searchTeam } from "../services/api";

const NAV = [
  { to: "/",               label: "Today" },
  { to: "/predictions",    label: "Picks" },
  { to: "/distributions",  label: "Distributions" },
  { to: "/results",        label: "Results" },
  { to: "/standings",      label: "Table" },
  { to: "/stats",          label: "Stats" },
];

function useTheme() {
  const [light, setLight] = useState(() => {
    try { return localStorage.getItem("theme") === "light"; } catch { return false; }
  });

  useEffect(() => {
    document.documentElement.classList.toggle("light", light);
    try { localStorage.setItem("theme", light ? "light" : "dark"); } catch {}
  }, [light]);

  return [light, () => setLight(v => !v)];
}

export default function Navbar() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [searching, setSearching] = useState(false);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const inputRef = useRef(null);
  const [isLight, toggleTheme] = useTheme();

  useEffect(() => {
    if (searching) inputRef.current?.focus();
  }, [searching]);

  async function submit(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setErr(null);
    try {
      const res = await searchTeam(query.trim());
      navigate(`/team/${res.data.team.id}`, { state: { team: res.data.team } });
      setSearching(false);
      setQuery("");
    } catch {
      setErr("Not found");
    } finally {
      setLoading(false);
    }
  }

  return (
    <header className="border-b border-zinc-800 sticky top-0 z-50 nav-bg-blur backdrop-blur-sm">
      <div className="max-w-3xl mx-auto px-4 h-12 flex items-center gap-4">
        {/* Logo */}
        <Link to="/" className="font-semibold text-white text-sm tracking-tight shrink-0">
          SoccerBet<span className="text-green-500">AI</span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex gap-1 flex-1">
          {NAV.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                pathname === l.to
                  ? "text-white bg-zinc-800"
                  : "text-zinc-500 hover:text-zinc-200"
              }`}
            >
              {l.label}
            </Link>
          ))}
        </nav>

        {/* Search + theme toggle */}
        <div className="flex items-center gap-2 flex-1 md:flex-none justify-end">
          {searching ? (
            <form onSubmit={submit} className="flex items-center gap-2 w-full md:w-56">
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => { setQuery(e.target.value); setErr(null); }}
                placeholder="Search team…"
                className="flex-1 bg-zinc-900 border border-zinc-700 text-white text-sm
                           rounded px-2.5 py-1.5 outline-none focus:border-zinc-500
                           placeholder-zinc-600 min-w-0"
              />
              <button type="submit" disabled={loading}
                className="text-xs text-zinc-400 hover:text-white shrink-0">
                {loading ? "…" : "Go"}
              </button>
              <button type="button" onClick={() => { setSearching(false); setErr(null); }}
                className="text-zinc-600 hover:text-zinc-300 text-xs shrink-0">✕</button>
            </form>
          ) : (
            <button onClick={() => setSearching(true)}
              className="text-zinc-500 hover:text-zinc-200 text-sm flex items-center gap-1.5">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
              </svg>
              <span className="hidden sm:block text-xs">Search</span>
            </button>
          )}

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            title={isLight ? "Switch to dark mode" : "Switch to light mode"}
            className="text-zinc-500 hover:text-zinc-200 transition-colors shrink-0 p-0.5"
          >
            {isLight ? (
              /* Moon icon — click to go dark */
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round"
                      d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
              </svg>
            ) : (
              /* Sun icon — click to go light */
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <circle cx="12" cy="12" r="5"/>
                <path strokeLinecap="round" strokeLinejoin="round"
                      d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42
                         M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Search error */}
      {err && (
        <div className="text-center text-xs text-red-400 py-1 border-t border-zinc-800">
          {err}
        </div>
      )}
    </header>
  );
}
