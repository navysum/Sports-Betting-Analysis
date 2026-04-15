export function kickoffTime(utcStr) {
  if (!utcStr) return "—";
  return new Date(utcStr).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}
