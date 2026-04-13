export default function StarsRating({ stars = 1, size = "sm" }) {
  const total = 5;
  const sz = size === "lg" ? "text-base" : "text-xs";
  return (
    <span className={`${sz} tracking-tight`} title={`${stars} / 5 stars`}>
      {Array.from({ length: total }, (_, i) => (
        <span key={i} className={i < stars ? "text-yellow-400" : "text-slate-600"}>
          ★
        </span>
      ))}
    </span>
  );
}
