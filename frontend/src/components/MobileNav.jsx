import { Link, useLocation } from "react-router-dom";

const TABS = [
  { to: "/",               label: "Today",     icon: "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" },
  { to: "/best-bets",      label: "Best Bets", icon: "M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" },
  { to: "/predictions",    label: "Picks",     icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" },
  { to: "/analytics",      label: "Analytics", icon: "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" },
  { to: "/standings",      label: "Table",     icon: "M4 6h16M4 10h16M4 14h16M4 18h16" },
];

export default function MobileNav() {
  const { pathname } = useLocation();
  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 nav-bg border-t border-zinc-800 flex"
         style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
      {TABS.map((t) => {
        const active = pathname === t.to;
        return (
          <Link key={t.to} to={t.to}
            className={`flex-1 flex flex-col items-center justify-center py-2 gap-0.5 min-h-[52px]
                        text-[10px] font-medium transition-colors
                        ${active ? "text-green-500" : "text-zinc-600"}`}>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24"
                 stroke="currentColor" strokeWidth={active ? 2 : 1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d={t.icon} />
            </svg>
            {t.label}
          </Link>
        );
      })}
    </nav>
  );
}
