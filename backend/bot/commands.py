"""
Telegram command handlers.

All handlers follow the pattern:
    async def cmd_xxx(update: Update, context: ContextTypes.DEFAULT_TYPE)
"""
import json
import os
import re
import asyncio
from datetime import datetime
from typing import Optional

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from app.config import settings
from app.services.football_api import (
    get_all_today_matches,
    get_all_tomorrow_matches,
    get_standings,
    get_team_matches,
    get_h2h,
    find_team_by_name,
    SUPPORTED_COMPETITIONS,
    FDORG_COMPETITIONS,
)
from app.services.prediction_service import predict_match, predict_upcoming_batch
from app.services.evaluator import get_accuracy_stats, get_accuracy_by_league
from ml.train import get_last_training_summary
from bot.formatter import (
    format_daily_briefing,
    format_match_analysis,
    format_form_guide,
    format_standings,
    format_h2h,
    format_bet_log,
    format_pnl,
    format_accuracy,
    format_backtest,
    format_saved_tips_list,
)

BETS_PATH       = os.path.join(settings.data_dir, "bets.json")
SAVED_TIPS_PATH = os.path.join(settings.data_dir, "saved_tips.json")
LEDGER_PATH     = os.path.join(settings.data_dir, "predictions.json")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_admin(update: Update) -> bool:
    return str(update.effective_user.id) == settings.telegram_user_id


async def _reply(update: Update, text: str) -> None:
    try:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception:
        # Fallback — strip markdown if it fails
        clean = re.sub(r"[_*`\[\]()~>#+=|{}.!\\]", "", text)
        await update.message.reply_text(clean)


def _load_bets() -> list[dict]:
    if not os.path.exists(BETS_PATH):
        return []
    try:
        with open(BETS_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_bets(bets: list[dict]) -> None:
    os.makedirs(os.path.dirname(BETS_PATH), exist_ok=True)
    with open(BETS_PATH, "w") as f:
        json.dump(bets, f, indent=2)


def _form_results(matches: list[dict], team_id: int) -> list[str]:
    from ml.features import _result_for_team
    finished = [m for m in matches if m.get("status") == "FINISHED"]
    return [_result_for_team(m, team_id) or "?" for m in finished[-10:]]


async def _batch_predictions_for_today(label: str = "Today") -> list[dict]:
    """Pull today's (or tomorrow's) matches and predict them."""
    getter = get_all_today_matches if label == "Today" else get_all_tomorrow_matches
    raw_matches = await getter()

    results = []
    for match in raw_matches:
        home_id = match.get("homeTeam", {}).get("id")
        away_id = match.get("awayTeam", {}).get("id")
        if not home_id or not away_id:
            continue

        home_name = match.get("homeTeam", {}).get("shortName") or match.get("homeTeam", {}).get("name", "")
        away_name = match.get("awayTeam", {}).get("shortName") or match.get("awayTeam", {}).get("name", "")
        code = match.get("_competition_code", "PL")

        try:
            pred = await predict_match(
                home_team_id=home_id,
                away_team_id=away_id,
                competition_code=code,
                api_match_id=match.get("id"),
                home_team_name=home_name,
                away_team_name=away_name,
                match_date=(match.get("utcDate") or "")[:10],
                save_to_ledger=True,
            )
        except Exception:
            pred = {}

        results.append({
            "competition": match.get("_competition_name", code),
            "match_date": match.get("utcDate"),
            "home_team": home_name,
            "away_team": away_name,
            "prediction": pred,
        })
        await asyncio.sleep(6)  # rate limit

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/today — Today's fixtures and predictions."""
    await update.message.reply_text("⏳ Fetching today's matches and running predictions…")
    try:
        predictions = await _batch_predictions_for_today("Today")
        date_str = datetime.now().strftime("%A %-d %B %Y")
        text = format_daily_briefing(predictions, label="Today", date_str=date_str)
    except Exception as e:
        text = f"❌ Error fetching today's fixtures: {e}"
    await _reply(update, text)


async def cmd_tomorrow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/tomorrow — Tomorrow's fixtures and predictions."""
    await update.message.reply_text("⏳ Fetching tomorrow's matches…")
    try:
        predictions = await _batch_predictions_for_today("Tomorrow")
        from datetime import timedelta
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%A %-d %B %Y")
        text = format_daily_briefing(predictions, label="Tomorrow", date_str=tomorrow)
    except Exception as e:
        text = f"❌ Error: {e}"
    await _reply(update, text)


async def cmd_analyse(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/analyse <Team A> vs <Team B>  or  /analyse <Team A> <Team B>"""
    args_text = " ".join(context.args) if context.args else ""

    # Parse: "Arsenal vs Chelsea" or "Arsenal Chelsea"
    if " vs " in args_text.lower():
        parts = re.split(r"\s+vs\s+", args_text, flags=re.IGNORECASE)
    elif len(context.args) >= 2:
        mid = len(context.args) // 2
        parts = [" ".join(context.args[:mid]), " ".join(context.args[mid:])]
    else:
        await _reply(update, "Usage: /analyse Arsenal vs Chelsea")
        return

    home_name = parts[0].strip()
    away_name = parts[1].strip() if len(parts) > 1 else ""

    if not home_name or not away_name:
        await _reply(update, "Please provide both team names\\.")
        return

    await update.message.reply_text(f"⏳ Analysing {home_name} vs {away_name}…")

    try:
        home_team = await find_team_by_name(home_name)
        away_team = await find_team_by_name(away_name)

        if not home_team:
            await _reply(update, f"❌ Team not found: {home_name}")
            return
        if not away_team:
            await _reply(update, f"❌ Team not found: {away_name}")
            return

        home_id = home_team["id"]
        away_id = away_team["id"]
        home_display = home_team.get("shortName") or home_team.get("name", home_name)
        away_display = away_team.get("shortName") or away_team.get("name", away_name)

        home_matches, away_matches = await asyncio.gather(
            get_team_matches(home_id, limit=25),
            get_team_matches(away_id, limit=25),
        )

        pred = await predict_match(
            home_team_id=home_id,
            away_team_id=away_id,
            home_team_name=home_display,
            away_team_name=away_display,
        )

        home_form = _form_results(home_matches, home_id)
        away_form = _form_results(away_matches, away_id)

        h2h_summary = ""
        h2h_wins = pred.get("h2h_home_rate", None)  # not directly available, left as text

        text = format_match_analysis(
            home_team=home_display,
            away_team=away_display,
            prediction=pred,
            home_form=home_form,
            away_form=away_form,
            h2h_summary="",
            key_factors=pred.get("key_factors", ""),
        )
    except Exception as e:
        text = f"❌ Analysis failed: {e}"

    await _reply(update, text)


# Short alias
async def cmd_a(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_analyse(update, context)


async def cmd_form(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/form <Team>"""
    if not context.args:
        await _reply(update, "Usage: /form Arsenal")
        return

    team_name = " ".join(context.args)
    await update.message.reply_text(f"⏳ Fetching form for {team_name}…")

    try:
        team = await find_team_by_name(team_name)
        if not team:
            await _reply(update, f"❌ Team not found: {team_name}")
            return
        team_id = team["id"]
        display = team.get("shortName") or team.get("name", team_name)
        matches = await get_team_matches(team_id, limit=15)
        text = format_form_guide(display, matches, team_id)
    except Exception as e:
        text = f"❌ Error: {e}"

    await _reply(update, text)


async def cmd_h2h(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/h2h <Team A> vs <Team B>"""
    args_text = " ".join(context.args) if context.args else ""
    if " vs " in args_text.lower():
        parts = re.split(r"\s+vs\s+", args_text, flags=re.IGNORECASE)
    elif len(context.args) >= 2:
        mid = len(context.args) // 2
        parts = [" ".join(context.args[:mid]), " ".join(context.args[mid:])]
    else:
        await _reply(update, "Usage: /h2h Arsenal vs Chelsea")
        return

    home_name = parts[0].strip()
    away_name = parts[1].strip() if len(parts) > 1 else ""

    await update.message.reply_text(f"⏳ Fetching H2H for {home_name} vs {away_name}…")

    try:
        home_team, away_team = await asyncio.gather(
            find_team_by_name(home_name),
            find_team_by_name(away_name),
        )
        if not home_team or not away_team:
            await _reply(update, "❌ One or both teams not found\\.")
            return

        home_display = home_team.get("shortName") or home_team.get("name", home_name)
        away_display = away_team.get("shortName") or away_team.get("name", away_name)

        # H2H via the team's recent matches (filter for meetings)
        home_matches = await get_team_matches(home_team["id"], limit=40)
        h2h = [
            m for m in home_matches
            if (
                m.get("homeTeam", {}).get("id") == away_team["id"]
                or m.get("awayTeam", {}).get("id") == away_team["id"]
            )
        ]

        text = format_h2h(home_display, away_display, h2h)
    except Exception as e:
        text = f"❌ Error: {e}"

    await _reply(update, text)


async def cmd_injuries(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/injuries <Team> — best-effort from available API data."""
    if not context.args:
        await _reply(update, "Usage: /injuries Arsenal")
        return

    team_name = " ".join(context.args)

    # API-Football free tier can give injury data; if key not configured, inform user
    if not settings.api_football_key:
        await _reply(update, (
            f"ℹ️ Injury data requires an API\\-Football key \\(RapidAPI free tier\\)\\.\n"
            f"Set `API_FOOTBALL_KEY` in your \\.env file\\.\n\n"
            f"_You can check {team_name}'s squad news at transfermarkt\\.com_"
        ))
        return

    await update.message.reply_text(f"⏳ Fetching injuries for {team_name}…")
    # TODO: implement API-Football /injuries endpoint when key is available
    await _reply(update, f"⚠️ Injury data not yet implemented\\. Check Transfermarkt for {team_name}\\.")


async def cmd_standings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/standings [League]  — defaults to Premier League."""
    league = " ".join(context.args).upper() if context.args else "PL"

    # Accept full names too
    name_map = {
        "PREMIER LEAGUE": "PL", "PL": "PL",
        "CHAMPIONSHIP": "ELC", "ELC": "ELC",
        "LA LIGA": "PD", "PD": "PD",
        "BUNDESLIGA": "BL1", "BL1": "BL1",
        "SERIE A": "SA", "SA": "SA",
        "LIGUE 1": "FL1", "FL1": "FL1",
        "CHAMPIONS LEAGUE": "CL", "CL": "CL",
        "EREDIVISIE": "DED", "DED": "DED",
        "PRIMERA LIGA": "PPL", "PPL": "PPL",
    }
    code = name_map.get(league, league)

    if code not in FDORG_COMPETITIONS:
        await _reply(update, f"❌ Unknown league: {league}\\. Try PL, BL1, SA, FL1, PD, ELC, DED\\.")
        return

    await update.message.reply_text("⏳ Fetching standings…")
    try:
        table = await get_standings(code)
        text = format_standings(table, SUPPORTED_COMPETITIONS.get(code, code))
    except Exception as e:
        text = f"❌ Error fetching standings: {e}"

    await _reply(update, text)


async def cmd_bet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/bet <Description> <Odds> <Stake>
    Example: /bet Arsenal ML 2.10 5
    """
    if len(context.args) < 3:
        await _reply(update, "Usage: /bet Arsenal ML 2\\.10 5")
        return

    # Last two args are odds and stake, everything before is the description
    try:
        stake = float(context.args[-1].replace("£", ""))
        odds  = float(context.args[-2])
        desc  = " ".join(context.args[:-2])
    except ValueError:
        await _reply(update, "❌ Could not parse odds/stake\\. Example: /bet Arsenal ML 2\\.10 5")
        return

    bets = _load_bets()
    new_bet = {
        "id": len(bets) + 1,
        "logged_at": datetime.utcnow().isoformat(),
        "description": desc,
        "odds": odds,
        "stake": stake,
        "status": "PENDING",
        "pnl": None,
        "settled_at": None,
    }
    bets.append(new_bet)
    _save_bets(bets)

    potential = round((odds - 1) * stake, 2)
    await _reply(update, (
        f"✅ *Bet logged \\#{new_bet['id']}*\n"
        f"  {desc} @ {odds} \\| Stake: £{stake:.2f}\n"
        f"  Potential profit: £{potential:.2f}"
    ))


async def cmd_settle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/settle <id> <won|lost|void>"""
    if len(context.args) < 2:
        await _reply(update, "Usage: /settle 3 won")
        return
    try:
        bet_id = int(context.args[0])
        outcome = context.args[1].upper()
        if outcome not in ("WON", "LOST", "VOID"):
            raise ValueError
    except ValueError:
        await _reply(update, "❌ Usage: /settle 3 won \\| /settle 3 lost")
        return

    bets = _load_bets()
    bet = next((b for b in bets if b["id"] == bet_id), None)
    if not bet:
        await _reply(update, f"❌ Bet \\#{bet_id} not found\\.")
        return

    bet["status"] = outcome
    bet["settled_at"] = datetime.utcnow().isoformat()
    if outcome == "WON":
        bet["pnl"] = round((bet["odds"] - 1) * bet["stake"], 2)
    elif outcome == "LOST":
        bet["pnl"] = -bet["stake"]
    else:
        bet["pnl"] = 0.0

    _save_bets(bets)
    pnl_str = f"\\+£{bet['pnl']:.2f}" if bet["pnl"] >= 0 else f"\\-£{abs(bet['pnl']):.2f}"
    await _reply(update, f"✅ Bet \\#{bet_id} settled as *{outcome}* \\| P&L: {pnl_str}")


async def cmd_bets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/bets — Show active (pending) bets."""
    bets = _load_bets()
    await _reply(update, format_bet_log(bets, active_only=True))


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/pnl — Full profit & loss summary."""
    bets = _load_bets()
    await _reply(update, format_pnl(bets))


async def cmd_accuracy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/accuracy — Model accuracy stats."""
    s7  = get_accuracy_stats(days=7)
    s30 = get_accuracy_stats(days=30)
    sAll = get_accuracy_stats()
    await _reply(update, format_accuracy(s7, s30, sAll))


async def cmd_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/retrain — Force a model retrain (admin only)."""
    if not _is_admin(update):
        await _reply(update, "❌ Admin only command\\.")
        return

    await update.message.reply_text("⏳ Starting model retrain — this will take several minutes…")

    try:
        import asyncio
        from ml.train import main as train_main
        await train_main()

        # Reload ELO and DC models in prediction service after retrain
        import app.services.prediction_service as ps
        from ml.elo import load_elo_ratings
        ps._elo = load_elo_ratings()

        # Reload DC model in predict module
        from ml.predict import load_model
        load_model()

        summary = get_last_training_summary()
        r_acc  = summary.get("result_model", {}).get("accuracy_mean", 0)
        r_ll   = summary.get("result_model", {}).get("log_loss", "?")
        dc     = summary.get("dixon_coles", {})
        dc_str = f"\n  DC teams: {dc.get('teams', '?')}" if dc else ""
        await _reply(update, (
            f"✅ *Retrain complete*\n"
            f"  Samples: {summary.get('samples', '?')}\n"
            f"  Result model CV: {r_acc:.1%}\n"
            f"  Log\\-loss: {r_ll}{dc_str}\n"
            f"  Completed: {summary.get('trained_at', '')[:16]}"
        ))
    except Exception as e:
        await _reply(update, f"❌ Retrain failed: {e}")


async def cmd_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/leagues — List all tracked leagues."""
    lines = ["*⚽ Tracked Leagues*", ""]
    for code, name in SUPPORTED_COMPETITIONS.items():
        available = "✅" if code in FDORG_COMPETITIONS else "🔜"
        lines.append(f"  {available} `{code}` — {name}")
    lines += ["", "_✅ = full data available \\| 🔜 = coming soon_"]
    await _reply(update, "\n".join(lines))


async def cmd_leaguestats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/leaguestats — Accuracy breakdown by league."""
    stats = get_accuracy_by_league()
    if not stats:
        await _reply(update, "_No settled predictions yet\\._")
        return

    lines = ["*📊 Accuracy by League*", ""]
    for league, s in stats.items():
        acc = f"{s['accuracy']:.0%}"
        lines.append(f"  *{_esc_cmd(league)}* — {s['correct']}/{s['total']} \\({_esc_cmd(acc)}\\)")

    await _reply(update, "\n".join(lines))


def _esc_cmd(text: str) -> str:
    """Minimal MarkdownV2 escape for inline use."""
    import re as _re
    return _re.sub(r"([_*\[\]()~`>#+=|{}.!\\-])", r"\\\1", str(text))


# ── Saved Tips helpers ─────────────────────────────────────────────────────────

def _load_saved_tips() -> list[dict]:
    if not os.path.exists(SAVED_TIPS_PATH):
        return []
    try:
        with open(SAVED_TIPS_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_saved_tips(tips: list[dict]) -> None:
    os.makedirs(os.path.dirname(SAVED_TIPS_PATH), exist_ok=True)
    with open(SAVED_TIPS_PATH, "w") as f:
        json.dump(tips, f, indent=2)


def _load_ledger() -> list[dict]:
    if not os.path.exists(LEDGER_PATH):
        return []
    try:
        with open(LEDGER_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _extract_vs_pairs(text: str) -> list[tuple[str, str]]:
    """
    Extract (home, away) team name pairs from a bot message's plain text.
    Operates line-by-line so team names never span across newlines.
    Matches "Arsenal vs Chelsea" and "Arsenal vs Chelsea · League" patterns.
    """
    # Uses [ \t] not \s so matches stay on one line
    pattern = re.compile(
        r'([A-Z][\w\'\.&](?:[\w\'\.& -]*[\w\'\.&])?)'  # home team
        r'[ \t]+vs[ \t]+'
        r'([A-Z][\w\'\.&](?:[\w\'\.& -]*[\w\'\.&])?)'  # away team
        r'(?=[ \t]*[·|\n]|[ \t]*$)',
    )
    pairs = []
    seen = set()
    for line in text.splitlines():
        for m in pattern.finditer(line):
            home = m.group(1).strip()
            away = m.group(2).strip()
            key = (home.lower(), away.lower())
            if 3 <= len(home) <= 40 and 3 <= len(away) <= 40 and key not in seen:
                pairs.append((home, away))
                seen.add(key)
    return pairs


def _find_prediction(home_query: str, away_query: str) -> Optional[dict]:
    """Search the prediction ledger for the most recent matching entry."""
    ledger = _load_ledger()
    if not ledger:
        return None

    hq = home_query.lower()
    aq = away_query.lower()

    candidates = []
    for entry in ledger:
        h = entry.get("home", "").lower()
        a = entry.get("away", "").lower()
        h_match = any(w in h for w in hq.split() if len(w) >= 3)
        a_match = any(w in a for w in aq.split() if len(w) >= 3)
        if h_match and a_match:
            candidates.append(entry)

    if not candidates:
        return None
    return sorted(candidates, key=lambda e: e.get("logged_at", ""), reverse=True)[0]


def _pin_prediction(entry: dict) -> bool:
    """
    Add a ledger entry to saved_tips.json.
    Returns True if added, False if already present.
    """
    tips = _load_saved_tips()
    if any(t.get("match_id") == entry["match_id"] for t in tips):
        return False
    tips.append({
        "match_id":    entry["match_id"],
        "saved_at":    datetime.utcnow().isoformat(),
        "date":        entry.get("date", ""),
        "home":        entry.get("home", ""),
        "away":        entry.get("away", ""),
        "competition": entry.get("league", ""),
        "prediction":  entry.get("prediction", {}),
    })
    _save_saved_tips(tips)
    return True


async def handle_save_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Reply handler — triggered when the user replies to any bot message with
    text containing "save prediction" (case insensitive).

    Parses all "X vs Y" pairs from the replied-to message and pins each
    matching prediction for the 23:00 nightly report.
    """
    replied = update.message.reply_to_message
    if not replied or not replied.text:
        await _reply(update, "❌ Reply to a prediction message to save it\\.")
        return

    pairs = _extract_vs_pairs(replied.text)
    if not pairs:
        await _reply(update, "❌ Couldn't find any matches in that message\\.")
        return

    saved_names = []
    already_names = []
    not_found_names = []

    for home_q, away_q in pairs:
        entry = _find_prediction(home_q, away_q)
        if not entry:
            not_found_names.append(f"{home_q} vs {away_q}")
            continue
        added = _pin_prediction(entry)
        label = f"{entry['home']} vs {entry['away']}"
        if added:
            pred   = entry.get("prediction", {})
            result = (pred.get("result") or "?").title()
            conf   = f"{pred.get('confidence', 0):.0%}"
            saved_names.append((label, result, conf))
        else:
            already_names.append(label)

    lines = []
    for label, result, conf in saved_names:
        lines.append(
            f"📌 *{_esc_cmd(label)}*  →  {_esc_cmd(result)} \\({_esc_cmd(conf)}\\)"
        )
    for label in already_names:
        lines.append(f"ℹ️ Already saved: {_esc_cmd(label)}")
    for label in not_found_names:
        lines.append(f"⚠️ Not in ledger: {_esc_cmd(label)}")

    if saved_names:
        lines.append("")
        lines.append("_Results at 23:00 BST\\._")

    if lines:
        await _reply(update, "\n".join(lines))
    else:
        await _reply(update, "❌ No predictions found to save\\.")


async def cmd_saved(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/saved — Show your currently saved predictions."""
    tips = _load_saved_tips()
    await _reply(update, format_saved_tips_list(tips))


async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/backtest — Simulate historical staking on FDCO data."""
    await update.message.reply_text("⏳ Running backtest on historical data…")
    try:
        from ml.backtest import run_backtest
        result = run_backtest()
        text = format_backtest(result)
    except Exception as e:
        text = f"❌ Backtest failed: {e}"
    await _reply(update, text)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help — Command reference."""
    text = (
        "*⚽ Sports Bet Analysis Bot*\n\n"
        "*Fixtures & Predictions*\n"
        "  /today — Today's fixtures \\& predictions\n"
        "  /tomorrow — Tomorrow's fixtures \\& predictions\n"
        "  /analyse Arsenal vs Chelsea — Deep analysis \\+ correct scores\n"
        "  /a Arsenal Chelsea — Short alias\n\n"
        "*Team Data*\n"
        "  /form Arsenal — Last 10 match form\n"
        "  /h2h Arsenal vs Chelsea — Head\\-to\\-head\n"
        "  /injuries Arsenal — Injury \\& suspension list\n"
        "  /standings \\[PL\\] — League table\n\n"
        "*Betting Tracker*\n"
        "  /bet Arsenal ML 2\\.10 5 — Log a bet\n"
        "  /settle 3 won — Settle bet \\#3\n"
        "  /bets — Active bets\n"
        "  /pnl — Profit \\& loss summary\n\n"
        "*Saved Tips*\n"
        "  Reply to any prediction with \"save prediction\" — Pin it for 23:00 review\n"
        "  /saved — Show your pinned predictions\n\n"
        "*Model*\n"
        "  /accuracy — Accuracy \\+ log\\-loss \\+ Brier score\n"
        "  /leaguestats — Accuracy breakdown by league\n"
        "  /backtest — Simulate historical staking \\(FDCO data\\)\n"
        "  /retrain — Force model retrain \\(admin\\)\n"
        "  /leagues — Tracked leagues list\n"
        "  /help — This message\n\n"
        "_Engine: XGBoost \\+ Dixon\\-Coles ensemble \\+ ELO \\+ bookmaker odds features_"
    )
    await _reply(update, text)
