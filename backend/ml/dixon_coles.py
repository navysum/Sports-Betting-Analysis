"""
Dixon-Coles Poisson model for football score prediction.

Improves on basic Poisson with two key additions:
  1. τ (tau) low-score correction — adjusts the probability of 0-0, 1-0, 0-1, 1-1
     scorelines which the basic Poisson model systematically misprices.
  2. Time decay — recent matches are weighted more heavily (half-life = 60 days),
     so the model reflects current team form rather than multi-season averages.

Parameters fitted via MLE (scipy L-BFGS-B) on FDCO historical match data.

Reference: Dixon & Coles (1997) "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market"

Usage:
    from ml.dixon_coles import load_dc_model
    dc = load_dc_model()
    if dc:
        info = dc.match_probs("Arsenal", "Chelsea")
        # → {"home": 0.48, "draw": 0.27, "away": 0.25,
        #    "over25": 0.60, "btts": 0.54,
        #    "xg_home": 1.71, "xg_away": 1.12,
        #    "correct_scores": [{"score": "1-1", "prob": 0.10}, ...]}
"""
import json
import math
import os
import numpy as np
from typing import Optional

try:
    from scipy.optimize import minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DC_PARAMS_PATH = os.path.join(_DATA_DIR, "dixon_coles_params.json")

HALF_LIFE_DAYS = 40.0    # Time decay: 40-day half-life (tighter recency bias)
MAX_GOALS = 8            # Truncate scoreline grid at 8 goals per side
DC_BLEND_WEIGHT = 0.50   # 50% Dixon-Coles, 50% XGBoost in blended output
MAX_MATCHES = 3000       # Cap matches used for MLE (keeps fitting fast)


# ─── τ correction ─────────────────────────────────────────────────────────────

def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score correction factor for scoreline (x, y)."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ─── Negative log-likelihood ──────────────────────────────────────────────────

def _neg_ll(
    params: np.ndarray,
    teams: list,
    matches: list,
    weights: np.ndarray,
    reg: float = 0.005,
) -> float:
    """Negative log-likelihood with L2 regularisation."""
    n = len(teams)
    attack  = {t: params[i]     for i, t in enumerate(teams)}
    defence = {t: params[n + i] for i, t in enumerate(teams)}
    home_adv = params[2 * n]
    rho      = params[2 * n + 1]

    ll = 0.0
    for i, m in enumerate(matches):
        h, a = m["home"], m["away"]
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        if h not in attack or a not in attack:
            continue
        lam = max(math.exp(attack[h]  + defence[a] + home_adv), 1e-6)
        mu  = max(math.exp(attack[a]  + defence[h]),             1e-6)
        t   = _tau(hg, ag, lam, mu, rho)
        if t <= 0:
            continue
        ll += weights[i] * (
            math.log(t)
            + hg * math.log(lam) - lam - math.lgamma(hg + 1)
            + ag * math.log(mu)  - mu  - math.lgamma(ag + 1)
        )

    # L2 regularisation on attack/defence (stabilises identification)
    reg_term = reg * float(np.sum(params[: 2 * n] ** 2))
    return -(ll - reg_term)


# ─── Model class ──────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Fitted Dixon-Coles model. After fitting, call match_probs(home, away)
    to get a full probability breakdown for a match.
    """

    def __init__(self):
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_adv: float = 0.3
        self.rho: float = -0.1
        self._teams: list = []
        self._fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        matches: list[dict],
        reference_date: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        Fit model parameters via MLE.

        matches: [{"home": str, "away": str, "home_goals": int,
                   "away_goals": int, "date": "YYYY-MM-DD"}, ...]
        """
        if not _SCIPY_AVAILABLE:
            print("  [dc] scipy not available — Dixon-Coles skipped.")
            return

        if not matches:
            return

        from datetime import datetime

        ref_dt = (
            datetime.fromisoformat(reference_date)
            if reference_date
            else datetime.now()
        )

        # Time-decay weights with season-start boost.
        # When ref_dt falls in Aug–Sep (start of new season), matches from
        # before July 1 of the same year are from the *previous* season and
        # should decay faster (2× rate) — team rosters/managers may have changed.
        season_start_month = ref_dt.month in (8, 9)

        weights = np.ones(len(matches), dtype=np.float64)
        for i, m in enumerate(matches):
            try:
                d = datetime.fromisoformat(m.get("date", "2020-01-01")[:10])
                days = max((ref_dt - d).days, 0)
                # Apply 2× decay rate for pre-summer matches at season start
                hl = HALF_LIFE_DAYS
                if season_start_month and d.year == ref_dt.year and d.month < 7:
                    hl = HALF_LIFE_DAYS / 2.0
                weights[i] = math.exp(-math.log(2) * days / hl)
            except Exception:
                weights[i] = 0.1

        teams = sorted(
            {m["home"] for m in matches} | {m["away"] for m in matches}
        )
        n = len(teams)
        self._teams = teams

        if n < 4:
            return

        # Initial params: attack=0.3, defence=0.0, home_adv=0.3, rho=-0.1
        x0 = np.concatenate([
            np.full(n, 0.3),    # attack
            np.zeros(n),        # defence
            [0.3],              # home_adv
            [-0.1],             # rho
        ])
        bounds = (
            [(-3.0, 3.0)] * n     # attack
            + [(-3.0, 3.0)] * n   # defence
            + [(0.0, 1.0)]        # home_adv
            + [(-0.99, 0.0)]      # rho (must be ≤ 0 for τ to be a valid correction)
        )

        if verbose:
            print(f"  [dc] Fitting Dixon-Coles on {len(matches)} matches, "
                  f"{n} teams…", flush=True)

        result = minimize(
            _neg_ll,
            x0,
            args=(teams, matches, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9, "maxfun": 50000},
        )

        params = result.x
        self.attack   = dict(zip(teams, params[:n]))
        self.defence  = dict(zip(teams, params[n: 2 * n]))
        self.home_adv = float(params[2 * n])
        self.rho      = float(params[2 * n + 1])
        self._fitted  = True

        if verbose:
            status = "converged" if result.success else f"stopped ({result.message})"
            print(f"  [dc] Fitting {status}. "
                  f"home_adv={self.home_adv:.3f}, rho={self.rho:.3f}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def _lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return (λ_home, μ_away) expected-goal estimates."""
        lam = math.exp(
            self.attack.get(home_team,  0.0)
            + self.defence.get(away_team, 0.0)
            + self.home_adv
        )
        mu = math.exp(
            self.attack.get(away_team,  0.0)
            + self.defence.get(home_team, 0.0)
        )
        return max(lam, 0.05), max(mu, 0.05)

    def scoreline_grid(
        self, home_team: str, away_team: str, max_goals: int = MAX_GOALS
    ) -> np.ndarray:
        """Return (max_goals+1 × max_goals+1) probability grid."""
        lam, mu = self._lambdas(home_team, away_team)

        # Poisson PMFs (computed once per λ/μ)
        lam_pmf = [
            math.exp(-lam) * (lam ** k) / math.factorial(k)
            for k in range(max_goals + 1)
        ]
        mu_pmf = [
            math.exp(-mu) * (mu ** k) / math.factorial(k)
            for k in range(max_goals + 1)
        ]

        grid = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                t = _tau(i, j, lam, mu, self.rho)
                grid[i, j] = lam_pmf[i] * mu_pmf[j] * t

        total = grid.sum()
        if total > 0:
            grid /= total
        return grid

    def match_probs(
        self, home_team: str, away_team: str
    ) -> Optional[dict]:
        """
        Returns a full probability breakdown, or None if neither team was
        in the training data (DC signal would be uninformative).
        """
        known_home = home_team in self.attack
        known_away = away_team in self.attack
        if not known_home and not known_away:
            return None

        grid = self.scoreline_grid(home_team, away_team)

        # 1X2
        home_win = float(np.sum(np.tril(grid, -1)))  # home_goals > away_goals
        draw     = float(np.trace(grid))
        away_win = float(np.sum(np.triu(grid, 1)))   # away_goals > home_goals

        # Over/under 2.5 goals
        under25 = sum(
            float(grid[i, j])
            for i in range(min(3, MAX_GOALS + 1))
            for j in range(min(3, MAX_GOALS + 1))
            if i + j <= 2
        )
        over25 = 1.0 - under25

        # Over/under 3.5 goals
        under35 = sum(
            float(grid[i, j])
            for i in range(min(4, MAX_GOALS + 1))
            for j in range(min(4, MAX_GOALS + 1))
            if i + j <= 3
        )
        over35 = 1.0 - under35

        # BTTS: P(home≥1) × P(away≥1) = 1 - P(home=0) - P(away=0) + P(0-0)
        btts = float(
            1.0 - grid[0, :].sum() - grid[:, 0].sum() + grid[0, 0]
        )

        # Negative correlation correction for high expected-goal scenarios.
        # The τ correction only adjusts {0-0, 1-0, 0-1, 1-1}; for all other
        # scorelines the model assumes goal independence. In reality, when one
        # team builds a large lead they slow the game, suppressing the opponent's
        # scoring threat. This overprices BTTS on attacking vs defensive mismatches.
        #
        # Empirical correction: reduce BTTS by ~2.5% per expected goal above 2.5
        # total xG. At lam+mu=4.0 this gives a −3.75% correction — in line with
        # what sharp models see in high-xG matchups.
        lam, mu = self._lambdas(home_team, away_team)
        xg_correction = max(0.0, (lam + mu - 2.5) * 0.025)
        btts = max(0.0, btts * (1.0 - xg_correction))

        # Top 12 correct-score probabilities (for display)
        flat = [
            (float(grid[i, j]), i, j)
            for i in range(grid.shape[0])
            for j in range(grid.shape[1])
        ]
        flat.sort(reverse=True)
        correct_scores = [
            {"score": f"{i}-{j}", "prob": round(p, 4)}
            for p, i, j in flat[:12]
        ]

        # Full score grid as nested list — used by frontend Monte Carlo to
        # sample directly from the τ-corrected distribution (far more accurate
        # than raw Poisson sampling, especially for low-scoring scorelines).
        score_grid = [
            [round(float(grid[i, j]), 6) for j in range(grid.shape[1])]
            for i in range(grid.shape[0])
        ]

        # lam / mu already computed above for the BTTS correction
        return {
            "home":           round(max(home_win, 0.0), 4),
            "draw":           round(max(draw,     0.0), 4),
            "away":           round(max(away_win, 0.0), 4),
            "over25":         round(max(min(over25, 1.0), 0.0), 4),
            "over35":         round(max(min(over35, 1.0), 0.0), 4),
            "btts":           round(max(min(btts,   1.0), 0.0), 4),
            "xg_home":        round(lam, 2),
            "xg_away":        round(mu,  2),
            "rho":            round(self.rho, 4),
            "correct_scores": correct_scores,
            "score_grid":     score_grid,
            "score_grid_size": grid.shape[0],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "attack":   self.attack,
            "defence":  self.defence,
            "home_adv": self.home_adv,
            "rho":      self.rho,
            "teams":    self._teams,
            "fitted":   self._fitted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DixonColesModel":
        dc = cls()
        dc.attack   = {k: float(v) for k, v in data.get("attack",  {}).items()}
        dc.defence  = {k: float(v) for k, v in data.get("defence", {}).items()}
        dc.home_adv = float(data.get("home_adv", 0.3))
        dc.rho      = float(data.get("rho",      -0.1))
        dc._teams   = data.get("teams", [])
        dc._fitted  = data.get("fitted", bool(dc.attack))
        return dc


# ─── I/O ──────────────────────────────────────────────────────────────────────

def save_dc_model(dc: DixonColesModel) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(DC_PARAMS_PATH, "w") as f:
        json.dump(dc.to_dict(), f, indent=2)
    n = len(dc.attack)
    print(f"  [dc] Dixon-Coles saved -> {DC_PARAMS_PATH} ({n} teams)")


def load_dc_model() -> Optional[DixonColesModel]:
    if not os.path.exists(DC_PARAMS_PATH):
        return None
    try:
        with open(DC_PARAMS_PATH) as f:
            data = json.load(f)
        dc = DixonColesModel.from_dict(data)
        if not dc._fitted or not dc.attack:
            return None
        return dc
    except Exception:
        return None
