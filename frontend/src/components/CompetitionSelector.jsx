export const COMPETITIONS = {
  PL:  "Premier League",
  PD:  "La Liga",
  BL1: "Bundesliga",
  SA:  "Serie A",
  FL1: "Ligue 1",
  CL:  "UCL",
  ELC: "Championship",
  DED: "Eredivisie",
  PPL: "Primeira Liga",
};

export const COMPETITION_FLAGS = {
  PL: "🏴󠁧󠁢󠁥󠁮󠁧󠁿", PD: "🇪🇸", BL1: "🇩🇪", SA: "🇮🇹",
  FL1: "🇫🇷", CL: "🏆", ELC: "🏴󠁧󠁢󠁥󠁮󠁧󠁿", DED: "🇳🇱", PPL: "🇵🇹",
};

/** Dropdown */
export default function CompetitionSelector({ value, onChange }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}
      className="bg-zinc-900 border border-zinc-700 text-zinc-200 rounded px-2.5 py-1.5
                 text-sm outline-none focus:border-zinc-500 cursor-pointer">
      {Object.entries(COMPETITIONS).map(([code, name]) => (
        <option key={code} value={code}>{COMPETITION_FLAGS[code]} {name}</option>
      ))}
    </select>
  );
}

/** Scrollable tab strip */
export function CompetitionTabs({ value, onChange }) {
  return (
    <div className="flex gap-0 overflow-x-auto no-scrollbar border-b border-zinc-800">
      {Object.entries(COMPETITIONS).map(([code, name]) => {
        const active = value === code;
        return (
          <button key={code} onClick={() => onChange(code)}
            className={`flex-shrink-0 px-3 py-2 text-xs font-medium border-b-2 transition-colors whitespace-nowrap
                        ${active
                          ? "border-green-500 text-white"
                          : "border-transparent text-zinc-500 hover:text-zinc-300"}`}>
            {name}
          </button>
        );
      })}
    </div>
  );
}
