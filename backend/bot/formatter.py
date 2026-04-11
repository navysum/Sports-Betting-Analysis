"""
Format data as clean Telegram messages (MarkdownV2).

All public functions return strings ready to pass to bot.send_message().
"""
from datetime import datetime
from typing import Optional
import pytz


TZ = pytz.timezone("Europe/London")
STAR_MAP = {1: "☆☆☆☆☆", 2: "★☆☆☆☆", 3: "★★☆☆☆", 4: "★★★☆☆", 5: "★★★★★"}

OUTCOME_EMOJI = {"HOME": "🏠", "DRAW": "🤝", "AWAY": "✈️"}
RESULT_EMOJI  = {True: "✅", False: "❌"}


def _esc(text: str) -> str:
    """Escape special MarkdownV2 chars."""
    for ch in r"\_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


def _pct(p: Optional[float]) -> str:
    if p is None:
        return "N/A"
    return f"{p:.0%}"


def _fmt_kickoff(utc_str: Optional[str]) -> str:
    if not utc_str:
        return "TBD"
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        local = dt.astimezone(TZ)
        return local.strftime("%H:%M")
    except Exception:
        return utc_str[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Daily Briefing
# ──────────────────────────────────────────────────────────────────────────────

def format_daily_briefing(
    predictions: list[dict],
    label: str = "Today",
    date_str: str = "",
) -> str:
    if not date_str:
        date_str = datetime.now(TZ).strftime("%A %-d %B %Y")

    if not predictions:
        return f"*{_esc(label)}'s Fixtures*\n\n_{_esc(date_str)}_\n\nNo matches found for tracked leagues\\."

    lines = [f"*⚽ {_esc(label)}'s Fixtures \\& Predictions*", f"_{_esc(date_str)}_", ""]

    by_comp: dict[str, list] = {}
    for item in predictions:
        comp = item.get("competition", "Unknown")
        by_comp.setdefault(comp, []).append(item)

    for comp, items in by_comp.items():
        lines.append(f"*{_esc(comp)}*")
        for item in items:
            pred = item.get("prediction", {})
            home = _esc(item.get("home_team", "?"))
            away = _esc(item.get("away_team", "?"))
            ko   = _fmt_kickoff(item.get("match_date"))
            outcome = pred.get("predicted_outcome", "?")
            stars  = STAR_MAP.get(pred.get("stars", 1), "")
            emoji  = OUTCOME_EMOJI.get(outcome, "")
            conf   = _pct(pred.get("confidence"))
            o25    = _pct(pred.get("over25_prob"))
            btts   = _pct(pred.get("btts_prob"))

            lines.append(
                f"  `{ko}` {home} vs {away}"
            )
            lines.append(
                f"  {emoji} {_esc(outcome.title())} {stars} \\({conf}\\)"
            )
            lines.append(
                f"  O2\\.5: {o25} \\| BTTS: {btts}"
            )

            # Value bets
            vbs = pred.get("value_bets", [])
            if vbs:
                lines.append(f"  💰 Value: {_esc(', '.join(vbs[:2]))}")

            lines.append("")

    lines.append("_Predictions are probabilistic — bet responsibly\\._")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Pre-Match Analysis
# ──────────────────────────────────────────────────────────────────────────────

def format_match_analysis(
    home_team: str,
    away_team: str,
    prediction: dict,
    home_form: list[str],    # e.g. ["W", "D", "W", "L", "W"]
    away_form: list[str],
    h2h_summary: str,
    key_factors: str,
    competition: str = "",
) -> str:
    home = _esc(home_team)
    away = _esc(away_team)
    comp = _esc(competition)
    outcome = prediction.get("predicted_outcome", "?")
    emoji  = OUTCOME_EMOJI.get(outcome, "")
    stars  = STAR_MAP.get(prediction.get("stars", 1), "")

    form_home = " ".join(_form_emoji(r) for r in home_form[-5:])
    form_away = " ".join(_form_emoji(r) for r in away_form[-5:])

    lines = [
        f"*🔍 Match Analysis*",
        f"*{home} vs {away}*" + (f" \\| {comp}" if comp else ""),
        "",
        f"*Prediction:* {emoji} {_esc(outcome.title())} {stars}",
        f"*Confidence:* {_pct(prediction.get('confidence'))}",
        f"*Over 2\\.5:* {_pct(prediction.get('over25_prob'))}",
        f"*BTTS:* {_pct(prediction.get('btts_prob'))}",
        "",
        f"*Form \\(last 5\\)*",
        f"  {home}: {form_home}",
        f"  {away}: {form_away}",
        "",
    ]

    if h2h_summary:
        lines += [f"*Head to Head*", f"  {_esc(h2h_summary)}", ""]

    if key_factors:
        lines += [f"*Key Factors*", f"  {_esc(key_factors)}", ""]

    vbs = prediction.get("value_bets", [])
    if vbs:
        lines.append(f"*💰 Value Bets*")
        for vb in vbs:
            lines.append(f"  • {_esc(vb)}")
        lines.append("")

    lines.append("_Predictions are probabilistic — bet responsibly\\._")
    return "\n".join(lines)


def _form_emoji(r: str) -> str:
    return {"W": "🟢", "D": "🟡", "L": "🔴"}.get(r.upper(), "⚪")


# ──────────────────────────────────────────────────────────────────────────────
# Form Guide
# ──────────────────────────────────────────────────────────────────────────────

def format_form_guide(team_name: str, matches: list[dict], team_id: int) -> str:
    from ml.features import _result_for_team
    team = _esc(team_name)
    lines = [f"*📋 Form Guide — {team}*", f"_Last {min(len(matches), 10)} matches_", ""]

    recent = [m for m in matches if m.get("status") == "FINISHED"][-10:]
    for m in reversed(recent):
        home  = m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name", "?")
        away  = m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name", "?")
        hg    = m.get("score", {}).get("fullTime", {}).get("home", "?")
        ag    = m.get("score", {}).get("fullTime", {}).get("away", "?")
        r     = _result_for_team(m, team_id)
        emoji = _form_emoji(r or "?")
        date  = (m.get("utcDate") or "")[:10]
        comp  = m.get("competition", {}).get("code", "")
        lines.append(f"  {emoji} `{date}` {_esc(home)} {hg}\\-{ag} {_esc(away)} `{_esc(comp)}`")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Standings
# ──────────────────────────────────────────────────────────────────────────────

def format_standings(table: list[dict], competition: str) -> str:
    comp = _esc(competition)
    lines = [f"*📊 {comp} Standings*", ""]
    lines.append("`# Team               P  W  D  L  GD Pts`")

    for row in table[:20]:
        pos  = row.get("position", "?")
        name = (row.get("team", {}).get("shortName") or row.get("team", {}).get("name", "?"))[:16]
        p    = row.get("playedGames", 0)
        w    = row.get("won", 0)
        d    = row.get("draw", 0)
        l    = row.get("lost", 0)
        gd   = row.get("goalDifference", 0)
        pts  = row.get("points", 0)
        gd_str = f"+{gd}" if gd > 0 else str(gd)
        lines.append(
            f"`{pos:>2} {name:<17} {p:>2} {w:>2} {d:>2} {l:>2} {gd_str:>4} {pts:>3}`"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# H2H
# ──────────────────────────────────────────────────────────────────────────────

def format_h2h(home_team: str, away_team: str, matches: list[dict]) -> str:
    home = _esc(home_team)
    away = _esc(away_team)
    lines = [f"*⚔️ Head to Head — {home} vs {away}*", ""]

    if not matches:
        return "\n".join(lines) + "_No H2H data available\\._"

    h_wins = d = a_wins = 0
    for m in matches[-10:]:
        h_id = m.get("homeTeam", {}).get("id")
        hg   = m.get("score", {}).get("fullTime", {}).get("home")
        ag   = m.get("score", {}).get("fullTime", {}).get("away")
        date = (m.get("utcDate") or "")[:10]
        mh   = m.get("homeTeam", {}).get("shortName") or m.get("homeTeam", {}).get("name", "?")
        ma   = m.get("awayTeam", {}).get("shortName") or m.get("awayTeam", {}).get("name", "?")
        if hg is None or ag is None:
            continue
        if hg > ag:   h_wins += 1
        elif hg == ag: d += 1
        else:          a_wins += 1
        lines.append(f"  `{date}` {_esc(mh)} {hg}\\-{ag} {_esc(ma)}")

    n = h_wins + d + a_wins or 1
    lines += [
        "",
        f"*Record \\(last {n}\\):*",
        f"  {home}: {h_wins} wins | {d} draws | {away}: {a_wins} wins",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Bet Log & P&L
# ──────────────────────────────────────────────────────────────────────────────

def format_bet_log(bets: list[dict], active_only: bool = False) -> str:
    title = "Active Bets" if active_only else "Bet Log"
    lines = [f"*🎰 {title}*", ""]

    if active_only:
        bets = [b for b in bets if b.get("status") == "PENDING"]

    if not bets:
        return "\n".join(lines) + "_No bets found\\._"

    for b in reversed(bets[-20:]):
        status = b.get("status", "PENDING")
        emoji  = {"PENDING": "⏳", "WON": "✅", "LOST": "❌", "VOID": "⚪"}.get(status, "❓")
        desc   = _esc(b.get("description", "?"))
        odds   = b.get("odds", 0)
        stake  = b.get("stake", 0)
        pnl_str = ""
        if b.get("pnl") is not None:
            pnl = b["pnl"]
            pnl_str = f" \\| {'\\+' if pnl >= 0 else ''}£{pnl:.2f}"
        lines.append(f"  {emoji} {desc} @ {_esc(str(odds))} \\| £{stake:.2f}{pnl_str}")

    return "\n".join(lines)


def format_pnl(bets: list[dict]) -> str:
    settled = [b for b in bets if b.get("status") in ("WON", "LOST", "VOID")]
    pending = [b for b in bets if b.get("status") == "PENDING"]

    total_staked = sum(b.get("stake", 0) for b in settled)
    total_pnl    = sum(b.get("pnl", 0) for b in settled if b.get("pnl") is not None)
    roi = (total_pnl / total_staked * 100) if total_staked else 0
    wins = sum(1 for b in settled if b.get("status") == "WON")
    losses = sum(1 for b in settled if b.get("status") == "LOST")
    strike = (wins / len(settled) * 100) if settled else 0

    pnl_sign = "\\+" if total_pnl >= 0 else ""
    roi_sign = "\\+" if roi >= 0 else ""

    lines = [
        "*📈 Profit & Loss Summary*",
        "",
        f"*Settled bets:* {len(settled)} \\({wins}W / {losses}L\\)",
        f"*Pending bets:* {len(pending)}",
        f"*Total staked:* £{total_staked:.2f}",
        f"*P&L:* {pnl_sign}£{total_pnl:.2f}",
        f"*ROI:* {roi_sign}{roi:.1f}%",
        f"*Strike rate:* {strike:.1f}%",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Accuracy Stats
# ──────────────────────────────────────────────────────────────────────────────

def format_accuracy(stats_7d: dict, stats_30d: dict, stats_all: dict) -> str:
    def _row(label: str, s: dict) -> str:
        if s["total"] == 0:
            return f"*{_esc(label)}:* _No data_"
        r_acc = _pct(s.get("result_accuracy"))
        o_acc = _pct(s.get("over25_accuracy")) if s.get("over25_accuracy") is not None else "N/A"
        b_acc = _pct(s.get("btts_accuracy"))   if s.get("btts_accuracy")   is not None else "N/A"
        return (
            f"*{_esc(label)}* \\({s['total']} predictions\\)\n"
            f"  Result: {r_acc} \\| Over2\\.5: {o_acc} \\| BTTS: {b_acc}"
        )

    lines = [
        "*📉 Model Accuracy*",
        "",
        _row("Last 7 days", stats_7d),
        "",
        _row("Last 30 days", stats_30d),
        "",
        _row("All time", stats_all),
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Weekly Report
# ──────────────────────────────────────────────────────────────────────────────

def format_backtest(result: dict) -> str:
    if "error" in result:
        return f"❌ {_esc(result['error'])}"

    flat   = result.get("flat", {})
    value  = result.get("value", {})
    kelly  = result.get("kelly", {})
    total  = result.get("total_matches", 0)

    def _pnl_str(pnl: float) -> str:
        sign = "\\+" if pnl >= 0 else ""
        return f"{sign}£{pnl:.2f}"

    def _roi_str(roi: float) -> str:
        sign = "\\+" if roi >= 0 else ""
        return f"{sign}{roi:.1f}%"

    lines = [
        "*📊 Historical Backtest \\(FDCO Data\\)*",
        f"_{total} matches analysed_",
        "",
        "*Flat staking \\(£1/match on predicted outcome\\)*",
        f"  Bets: {flat.get('bets', 0)} \\| Wins: {flat.get('wins', 0)} \\({flat.get('win_rate', 0):.1f}%\\)",
        f"  P&L: {_pnl_str(flat.get('pnl', 0))} \\| ROI: {_roi_str(flat.get('roi', 0))}",
        "",
        f"*Value staking \\(edge ≥ {value.get('min_edge_pct', 3)}%, £1/bet\\)*",
        f"  Bets: {value.get('bets', 0)} \\| Wins: {value.get('wins', 0)} \\({value.get('win_rate', 0):.1f}%\\)",
        f"  P&L: {_pnl_str(value.get('pnl', 0))} \\| ROI: {_roi_str(value.get('roi', 0))}",
        "",
        f"*Kelly staking \\({int(kelly.get('fraction', 0.25)*100)}% Kelly on value bets\\)*",
        f"  Bets: {kelly.get('bets', 0)} \\| Wins: {kelly.get('wins', 0)} \\({kelly.get('win_rate', 0):.1f}%\\)",
        f"  Bankroll: £{kelly.get('starting_bankroll', 100):.0f} → £{kelly.get('final_bankroll', 0):.2f}",
        f"  Max drawdown: {kelly.get('max_drawdown_pct', 0):.1f}%",
        "",
        "_Backtest uses in\\-sample data — live performance may differ\\._",
    ]
    return "\n".join(lines)


def format_weekly_report(
    accuracy_7d: dict,
    bets: list[dict],
    training_log: dict,
) -> str:
    acc = accuracy_7d
    settled = [b for b in bets if b.get("status") in ("WON", "LOST")]
    pnl = sum(b.get("pnl", 0) for b in settled)
    pnl_sign = "\\+" if pnl >= 0 else ""

    lines = [
        "*📋 Weekly Performance Report*",
        f"_Week ending {datetime.now(TZ).strftime('%A %-d %B %Y')}_",
        "",
        "*Model Accuracy \\(last 7 days\\)*",
    ]
    if acc["total"]:
        lines.append(
            f"  {acc['correct_result']}/{acc['total']} correct "
            f"\\({_pct(acc.get('result_accuracy'))}\\)"
        )
    else:
        lines.append("  _No settled predictions this week_")

    lines += [
        "",
        f"*Betting P&L \\(this week\\):* {pnl_sign}£{pnl:.2f}",
        "",
    ]

    if training_log:
        r_acc = training_log.get("result_model", {}).get("accuracy_mean", 0)
        lines += [
            f"*Last Retrain \\({_esc(training_log.get('trained_at', '')[:10])}\\)*",
            f"  Samples: {training_log.get('samples', '?')}",
            f"  Result model CV: {_pct(r_acc)}",
        ]

    return "\n".join(lines)
