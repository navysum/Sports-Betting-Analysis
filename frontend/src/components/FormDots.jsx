const DOT = {
  W: { cls: "bg-green-500", title: "Win" },
  D: { cls: "bg-yellow-400", title: "Draw" },
  L: { cls: "bg-red-500", title: "Loss" },
};

export default function FormDots({ form = [] }) {
  if (!form.length) return null;
  return (
    <div className="flex gap-1 items-center">
      {form.map((r, i) => {
        const { cls, title } = DOT[r?.toUpperCase()] || { cls: "bg-slate-600", title: "?" };
        return (
          <span
            key={i}
            className={`w-2.5 h-2.5 rounded-full ${cls}`}
            title={title}
          />
        );
      })}
    </div>
  );
}
