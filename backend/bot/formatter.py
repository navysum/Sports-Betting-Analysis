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

def _is_flat_prediction(pred: dict) -> bool:
    """
    Returns True if the model had insufficient data and returned near-uniform
    probabilities (all three outcomes within 8pp of each other).
    These are not useful predictions and should be hidden.
    """
    h = pred.get("home_win_prob", 0)
    d = pred.get("draw_prob", 0)
    a = pred.get("away_win_prob", 0)
    spread = max(h, d, a) - min(h, d, a)
    return spread < 0.08


def _fmt_value_bets(value_bets: list[str]) -> str:
    """Compact value bet display: 'Home Win +8%' instead of the full verbose string."""
    out = []
    for vb in value_bets[:3]:
        # Parse "Home Win (model 48% vs implied 20%)" → "Home Win +28%"
        try:
            label = vb.split("(")[0].strip()
            import re
            nums = re.findall(r"(\d+)%", vb)
            if len(nums) >= 2:
                edge = int(nums[0]) - int(nums[1])
                out.append(f"{label} \\+{edge}%")
            else:
                out.append(_esc(label))
        except Exception:
            out.append(_esc(vb[:30]))
    return " \\| ".join(out)


def _fmt_match_block(item: dict, show_comp: bool = True) -> list[str]:
    """Format a single match as a compact block of lines."""
    pred = item.get("prediction", {})
    home = _esc(item.get("home_team", "?"))
    away = _esc(item.get("away_team", "?"))
    ko   = _fmt_kickoff(item.get("match_date"))
    comp = _esc(item.get("competition", ""))

    outcome = pred.get("predicted_outcome", "?")
    stars   = STAR_MAP.get(pred.get("stars", 1), "")
    emoji   = OUTCOME_EMOJI.get(outcome, "")
    conf    = _pct(pred.get("confidence"))
    o25     = _pct(pred.get("over25_prob"))
    btts    = _pct(pred.get("btts_prob"))

    hp = _pct(pred.get("home_win_prob"))
    dp = _pct(pred.get("draw_prob"))
    ap = _pct(pred.get("away_win_prob"))

    comp_str = f" · {comp}" if show_comp and comp else ""
    lines = [
        f"*{home} vs {away}*{comp_str} · `{ko}`",
        f"{emoji} *{_esc(outcome.title())}* {stars} {conf}  \\|  H:{hp} D:{dp} A:{ap}",
        f"O2\\.5: {o25}  \\|  BTTS: {btts}",
    ]

    vbs = pred.get("value_bets", [])
    if vbs:
        lines.append(f"💰 {_fmt_value_bets(vbs)}")

    return lines


def format_daily_briefing(
    predictions: list[dict],
    label: str = "Today",
    date_str: str = "",
) -> str:
    if not date_str:
        date_str = datetime.now(TZ).strftime("%A %-d %B %Y")

    if not predictions:
        return f"*{_esc(label)}'s Fixtures*\n\n_{_esc(date_str)}_\n\nNo matches found for tracked leagues\\."

    # Filter out flat/no-data predictions and sort by confidence descending
    valid = [p for p in predictions if not _is_flat_prediction(p.get("prediction", {}))]
    flat  = [p for p in predictions if _is_flat_prediction(p.get("prediction", {}))]
    valid.sort(key=lambda p: p.get("prediction", {}).get("confidence", 0), reverse=True)

    lines = [
        f"*⚽ {_esc(label)}'s Predictions*",
        f"_{_esc(date_str)}_",
        "",
    ]

    # ── Strong picks (4–5 stars, ≥60% confidence) ──────────────────────────
    strong = [p for p in valid if p.get("prediction", {}).get("stars", 0) >= 4]
    if strong:
        lines.append("*🔥 Strong Picks*")
        lines.append("─────────────────")
        for item in strong:
            lines += _fmt_match_block(item, show_comp=True)
            lines.append("")

    # ── Moderate picks (3 stars, 50–59%) ────────────────────────────────────
    moderate = [p for p in valid if p.get("prediction", {}).get("stars", 0) == 3]
    if moderate:
        lines.append("*✅ Moderate Picks*")
        lines.append("─────────────────")
        for item in moderate:
            lines += _fmt_match_block(item, show_comp=True)
            lines.append("")

    # ── Low confidence (1–2 stars, <50%) ────────────────────────────────────
    low = [p for p in valid if p.get("prediction", {}).get("stars", 0) <= 2]
    if low:
        lines.append("*📋 Other Fixtures*")
        lines.append("─────────────────")
        for item in low:
            lines += _fmt_match_block(item, show_comp=True)
            lines.append("")

    # ── Matches with insufficient data ──────────────────────────────────────
    if flat:
        team_list = ", ".join(
            _esc(f"{p.get('home_team','?')} vs {p.get('away_team','?')}")
            for p in flat[:6]
        )
        more = f" \\+{len(flat)-6} more" if len(flat) > 6 else ""
        lines.append(f"_⚠️ {len(flat)} matches skipped \\(insufficient historical data\\): {team_list}{more}_")
        lines.append("")

    lines.append("_Predictions sorted by confidence — bet responsibly\\._")
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

    hp = _pct(prediction.get("home_win_prob"))
    dp = _pct(prediction.get("draw_prob"))
    ap = _pct(prediction.get("away_win_prob"))

    lines = [
        f"*🔍 Match Analysis*",
        f"*{home} vs {away}*" + (f" \\| {comp}" if comp else ""),
        "",
        f"*Prediction:* {emoji} {_esc(outcome.title())} {stars}",
        f"*H / D / A:*   {hp} \\| {dp} \\| {ap}",
        f"*Over 2\\.5:* {_pct(prediction.get('over25_prob'))}  \\|  *BTTS:* {_pct(prediction.get('btts_prob'))}",
        "",
    ]

    # Dixon-Coles expected goals + correct scores
    xg_home = prediction.get("xg_home")
    xg_away = prediction.get("xg_away")
    if xg_home is not None and xg_away is not None:
        lines.append(
            f"*xG \\(DC model\\):* {home} {_esc(str(round(xg_home, 2)))} "
            f"\\| {away} {_esc(str(round(xg_away, 2)))}"
        )
    correct_scores = prediction.get("correct_scores", [])
    if correct_scores:
        top_parts = []
        for s in correct_scores[:4]:
            pct = _esc(f"{s['prob']:.0%}")
            top_parts.append(f"`{s['score']}` {pct}")
        lines.append(f"*🎯 Likely Scores:* {'  '.join(top_parts)}")

    lines += [
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
        ll    = s.get("log_loss")
        brier = s.get("brier_score")
        ll_str    = _esc(f"{ll:.3f}")    if ll    is not None else "N/A"
        brier_str = _esc(f"{brier:.3f}") if brier is not None else "N/A"
        return (
            f"*{_esc(label)}* \\({s['total']} predictions\\)\n"
            f"  Result: {r_acc} \\| Over2\\.5: {o_acc} \\| BTTS: {b_acc}\n"
            f"  Log\\-loss: {ll_str} \\| Brier: {brier_str}"
        )

    lines = [
        "*📉 Model Accuracy*",
        "",
        _row("Last 7 days", stats_7d),
        "",
        _row("Last 30 days", stats_30d),
        "",
        _row("All time", stats_all),
        "",
        "_Log\\-loss: lower = better \\(random ≈ 1\\.10\\)_",
        "_Brier: lower = better \\(random ≈ 0\\.67\\)_",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Weekly Report
# ──────────────────────────────────────────────────────────────────────────────

def format_backtest(result: dict) -> str:
    if "error" in result:
        return f"❌ {_esc(result['error'])}"

    flat    = result.get("flat", {})
    value   = result.get("value", {})
    kelly   = result.get("kelly", {})
    total   = result.get("total_matches", 0)
    holdout = result.get("holdout_pct", 30)

    def _n(v, decimals=1):
        """Format a number and escape it for MarkdownV2."""
        return _esc(f"{v:.{decimals}f}")

    def _pnl(pnl: float) -> str:
        sign = "+" if pnl >= 0 else ""
        return _esc(f"{sign}£{pnl:.2f}")

    def _roi(roi: float) -> str:
        sign = "+" if roi >= 0 else ""
        return _esc(f"{sign}{roi:.1f}%")

    def _wr(wr: float) -> str:
        return _esc(f"{wr:.1f}%")

    lines = [
        "*📊 Backtest \\(most recent {}% of data\\)*".format(holdout),
        f"_{_esc(str(total))} matches in holdout_",
        "",
        "*Flat — £1 on predicted outcome every match*",
        f"  Bets: {flat.get('bets', 0)}  \\|  Win rate: {_wr(flat.get('win_rate', 0))}",
        f"  P&L: {_pnl(flat.get('pnl', 0))}  \\|  ROI: {_roi(flat.get('roi', 0))}",
        "",
        f"*Value — bet only when model edge ≥ {_esc(str(value.get('min_edge_pct', 3)))}%*",
        f"  Bets: {value.get('bets', 0)}  \\|  Win rate: {_wr(value.get('win_rate', 0))}",
        f"  P&L: {_pnl(value.get('pnl', 0))}  \\|  ROI: {_roi(value.get('roi', 0))}",
        "",
        f"*Kelly — {int(kelly.get('fraction', 0.25)*100)}% Kelly staking on value bets*",
        f"  Bets: {kelly.get('bets', 0)}  \\|  Win rate: {_wr(kelly.get('win_rate', 0))}",
        f"  Bankroll: {_esc('£' + str(int(kelly.get('starting_bankroll', 100))))} → {_pnl(kelly.get('final_bankroll', 0))}",
        f"  Max drawdown: {_esc(str(kelly.get('max_drawdown_pct', 0)) + '%')}",
        "",
        "_Note: model was trained on older seasons — holdout is partially in\\-sample\\._",
        "_For fully out\\-of\\-sample results, track live predictions with /accuracy\\._",
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
