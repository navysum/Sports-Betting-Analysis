"""
Dixon-Coles Poisson / Negative-Binomial score-prediction model.

Improvements over the original:
  1. τ (tau) low-score correction — adjusts {0-0, 1-0, 0-1, 1-1} probabilities.
  2. Time decay — 40-day half-life; pre-season matches decay 2× faster.
  3. FIX #8  — Per-league rho: one τ-correction strength per competition.
               FL1 and ELC produce far more 0-0 draws than BL1; a single
               global rho was misspecified for every league.
  4. FIX #9  — Convergence retry: if L-BFGS-B does not converge, the
               optimiser retries with a lightly perturbed starting point and
               keeps whichever solution has the lower negative log-likelihood.
  5. FIX #5  — BTTS xG correction removed.  The previous post-hoc adjustment
               made the headline BTTS% inconsistent with what the Monte Carlo
               simulator computed directly from the score grid.  BTTS is now
               derived purely from the grid so both figures always agree.
  6. Negative Binomial (NB) scoring model — fitted alongside Poisson.
               NB has an additional dispersion parameter r; as r→∞ it reduces
               to Poisson. Football goals are mildly overdispersed relative to
               Poisson (more 0-0 and 5+ games than pure Poisson predicts); NB
               captures this without over-correcting. The model fits r jointly
               with attack/defence parameters and uses NB PMFs in the score
               grid.

Reference: Dixon & Coles (1997) "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market"

Usage:
    from ml.dixon_coles import load_dc_model
    dc = load_dc_model()
    if dc:
        info = dc.match_probs("Arsenal", "Chelsea", league="PL")
        # → {"home": 0.48, "draw": 0.27, "away": 0.25,
        #    "over25": 0.60, "btts": 0.54, "over35": 0.32,
        #    "xg_home": 1.71, "xg_away": 1.12,
        #    "rho": -0.06,   # league-specific
        #    "correct_scores": [...], "score_grid": [...]}
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

HALF_LIFE_DAYS  = 40.0    # Time decay half-life in days
MAX_GOALS       = 8       # Truncate score grid at 8 goals per side
DC_BLEND_WEIGHT = 0.50    # Default DC weight in blended output (overridden by optimiser)
MAX_MATCHES     = 3000    # Cap matches fed to MLE (keeps fitting fast)
_NB_R_MIN       = 2.0     # Minimum NB dispersion (very overdispersed)
_NB_R_MAX       = 200.0   # Maximum NB dispersion (≈ Poisson at upper end)
_NB_R_INIT      = 20.0    # Initial value — mild overdispersion


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


# ─── Negative Binomial PMF ────────────────────────────────────────────────────

def _nb_pmf(k: int, mu: float, r: float) -> float:
    """
    Negative Binomial PMF (NB2 parameterisation).

      P(X=k | μ, r) = Γ(k+r) / (Γ(r) · k!) · (r/(r+μ))^r · (μ/(r+μ))^k

    r → ∞  recovers the Poisson(μ) PMF.
    r small = more overdispersion (heavier tails than Poisson).
    """
    if r <= 0 or mu <= 0:
        return math.exp(-mu) * (mu ** k) / math.factorial(k)   # fallback to Poisson
    log_pmf = (
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(r / (r + mu))
        + k * math.log(mu / (r + mu))
    )
    return math.exp(log_pmf)


# ─── Negative log-likelihood ──────────────────────────────────────────────────

def _neg_ll(
    params: np.ndarray,
    teams: list,
    matches: list,
    weights: np.ndarray,
    league_list: list,       # FIX #8: ordered unique league codes
    reg: float = 0.005,
) -> float:
    """
    Negative log-likelihood with L2 regularisation.

    Parameter layout (length = 2n + 2 + L):
      params[0   : n]        attack strengths
      params[n   : 2n]       defence strengths
      params[2n]             home advantage (log scale)
      params[2n+1]           NB dispersion r  (fitted in log space internally)
      params[2n+2 : 2n+2+L]  per-league rho values (FIX #8)
    """
    n = len(teams)
    L = len(league_list)

    attack   = {t: params[i]     for i, t in enumerate(teams)}
    defence  = {t: params[n + i] for i, t in enumerate(teams)}
    home_adv = params[2 * n]
    r_nb     = max(params[2 * n + 1], _NB_R_MIN)   # clamp to avoid degenerate NB
    rho_map  = {
        lg: params[2 * n + 2 + j]
        for j, lg in enumerate(league_list)
    }

    ll = 0.0
    for i, m in enumerate(matches):
        h, a = m["home"], m["away"]
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        league = m.get("league", league_list[0] if league_list else "default")
        if h not in attack or a not in attack:
            continue

        lam = max(math.exp(attack[h]  + defence[a] + home_adv), 1e-6)
        mu  = max(math.exp(attack[a]  + defence[h]),             1e-6)
        rho = rho_map.get(league, -0.1)
        t   = _tau(hg, ag, lam, mu, rho)
        if t <= 0:
            continue

        log_p_home = (
            math.lgamma(hg + r_nb) - math.lgamma(r_nb) - math.lgamma(hg + 1)
            + r_nb * math.log(r_nb / (r_nb + lam))
            + hg  * math.log(lam  / (r_nb + lam))
        )
        log_p_away = (
            math.lgamma(ag + r_nb) - math.lgamma(r_nb) - math.lgamma(ag + 1)
            + r_nb * math.log(r_nb / (r_nb + mu))
            + ag  * math.log(mu   / (r_nb + mu))
        )

        ll += weights[i] * (math.log(t) + log_p_home + log_p_away)

    # L2 regularisation on attack/defence (stabilises identification)
    reg_term = reg * float(np.sum(params[: 2 * n] ** 2))
    return -(ll - reg_term)


# ─── Model class ──────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Fitted Dixon-Coles model with per-league rho and Negative Binomial scoring.

    After fitting, call match_probs(home, away, league=code) to get a full
    probability breakdown for a match.
    """

    def __init__(self):
        self.attack: dict[str, float]       = {}
        self.defence: dict[str, float]      = {}
        self.home_adv: float                = 0.3
        self.rho: float                     = -0.1   # global fallback
        self.rho_by_league: dict[str, float]= {}     # FIX #8: per-league rho
        self.r_nb: float                    = _NB_R_INIT  # NB dispersion parameter
        self._teams: list                   = []
        self._fitted: bool                  = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_rho(self, league: str = "") -> float:
        """Return the τ-correction rho for a given league (falls back to global)."""
        return self.rho_by_league.get(league, self.rho)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        matches: list[dict],
        reference_date: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        Fit model parameters via MLE (L-BFGS-B).

        matches: [{"home": str, "away": str, "home_goals": int,
                   "away_goals": int, "date": "YYYY-MM-DD",
                   "league": "PL"}  ← FIX #8: league field required for per-league rho
                  ...]
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

        # Time-decay weights with season-start boost
        season_start_month = ref_dt.month in (8, 9)
        weights = np.ones(len(matches), dtype=np.float64)
        for i, m in enumerate(matches):
            try:
                d = datetime.fromisoformat(m.get("date", "2020-01-01")[:10])
                days = max((ref_dt - d).days, 0)
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

        # FIX #8: extract unique leagues to fit per-league rho
        league_list = sorted({m.get("league", "default") for m in matches})
        L = len(league_list)

        if verbose:
            print(
                f"  [dc] Fitting on {len(matches)} matches, {n} teams, "
                f"{L} leagues (NB dispersion + per-league rho)…", flush=True
            )

        # Parameter vector: [attack×n, defence×n, home_adv, r_nb, rho×L]
        x0 = np.concatenate([
            np.full(n, 0.3),           # attack
            np.zeros(n),               # defence
            [0.3],                     # home_adv
            [_NB_R_INIT],              # NB dispersion r
            [-0.1] * L,               # per-league rho (FIX #8)
        ])
        bounds = (
            [(-3.0,  3.0)] * n         # attack
            + [(-3.0, 3.0)] * n        # defence
            + [(0.0,  1.0)]            # home_adv
            + [(_NB_R_MIN, _NB_R_MAX)] # NB dispersion r
            + [(-0.99, 0.0)] * L       # per-league rho (must be ≤0 for valid τ)
        )

        opt_args = (teams, matches, weights, league_list)
        opt_kwargs = dict(
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9, "maxfun": 50000},
        )

        result = minimize(_neg_ll, x0, args=opt_args, **opt_kwargs)

        # FIX #9: convergence retry — if L-BFGS-B did not converge, retry
        # with a small perturbation from the best point found so far and keep
        # whichever run achieves the lower negative log-likelihood.
        if not result.success:
            if verbose:
                print("  [dc] First pass did not fully converge — retrying "
                      "with perturbed start…", flush=True)
            rng = np.random.default_rng(seed=42)
            x1 = result.x + rng.normal(0, 0.05, len(result.x))
            # Clip to bounds
            for k, (lo, hi) in enumerate(bounds):
                x1[k] = max(lo, min(hi, x1[k]))
            result2 = minimize(_neg_ll, x1, args=opt_args, **opt_kwargs)
            if result2.fun < result.fun:
                result = result2
                if verbose:
                    print(f"  [dc] Retry improved NLL: {result.fun:.4f}", flush=True)

        params = result.x
        self.attack   = dict(zip(teams, params[:n]))
        self.defence  = dict(zip(teams, params[n: 2 * n]))
        self.home_adv = float(params[2 * n])
        self.r_nb     = float(max(params[2 * n + 1], _NB_R_MIN))
        self.rho_by_league = {
            lg: float(params[2 * n + 2 + j])
            for j, lg in enumerate(league_list)
        }
        # Global rho = average across leagues (used as fallback for unknown leagues)
        self.rho = float(np.mean(list(self.rho_by_league.values()))) \
                   if self.rho_by_league else -0.1
        self._fitted = True

        if verbose:
            status = "converged" if result.success else f"stopped ({result.message})"
            print(
                f"  [dc] {status}. home_adv={self.home_adv:.3f}, "
                f"r_nb={self.r_nb:.1f}, "
                f"rho_by_league={{{', '.join(f'{k}: {v:.3f}' for k, v in self.rho_by_league.items())}}}"
            )

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
        self,
        home_team: str,
        away_team: str,
        max_goals: int = MAX_GOALS,
        league: str = "",
    ) -> np.ndarray:
        """
        Return (max_goals+1 × max_goals+1) probability grid.

        Uses Negative Binomial PMFs (with fitted dispersion r_nb) rather than
        plain Poisson. The τ correction uses the league-specific rho (FIX #8).
        """
        lam, mu = self._lambdas(home_team, away_team)
        rho = self._get_rho(league)
        r   = self.r_nb

        # NB PMFs for each goal count
        lam_pmf = [_nb_pmf(k, lam, r) for k in range(max_goals + 1)]
        mu_pmf  = [_nb_pmf(k, mu,  r) for k in range(max_goals + 1)]

        grid = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                t = _tau(i, j, lam, mu, rho)
                grid[i, j] = lam_pmf[i] * mu_pmf[j] * t

        total = grid.sum()
        if total > 0:
            grid /= total
        return grid

    def match_probs(
        self,
        home_team: str,
        away_team: str,
        league: str = "",
    ) -> Optional[dict]:
        """
        Returns a full probability breakdown for the match, or None if neither
        team was in the training data (DC signal would be uninformative).

        FIX #5: BTTS is now derived purely from the score grid so it is
        mathematically consistent with what the Monte Carlo simulator computes.
        The previous post-hoc xG correction caused a mismatch between the
        headline BTTS% and the MC result.

        FIX #8: league parameter selects the per-competition rho for the
        τ correction.
        """
        known_home = home_team in self.attack
        known_away = away_team in self.attack
        if not known_home and not known_away:
            return None

        grid = self.scoreline_grid(home_team, away_team, league=league)

        # 1X2
        home_win = float(np.sum(np.tril(grid, -1)))
        draw     = float(np.trace(grid))
        away_win = float(np.sum(np.triu(grid, 1)))

        # Over/Under 2.5
        under25 = sum(
            float(grid[i, j])
            for i in range(min(3, MAX_GOALS + 1))
            for j in range(min(3, MAX_GOALS + 1))
            if i + j <= 2
        )
        over25 = 1.0 - under25

        # Over/Under 3.5
        under35 = sum(
            float(grid[i, j])
            for i in range(min(4, MAX_GOALS + 1))
            for j in range(min(4, MAX_GOALS + 1))
            if i + j <= 3
        )
        over35 = 1.0 - under35

        # BTTS: derived directly from grid (FIX #5 — no post-hoc xG correction)
        # P(home ≥ 1 AND away ≥ 1) = 1 - P(home=0) - P(away=0) + P(0-0)
        btts = float(
            1.0 - grid[0, :].sum() - grid[:, 0].sum() + grid[0, 0]
        )

        lam, mu = self._lambdas(home_team, away_team)

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

        # Full score grid as nested list — used by the Monte Carlo simulator
        score_grid = [
            [round(float(grid[i, j]), 6) for j in range(grid.shape[1])]
            for i in range(grid.shape[0])
        ]

        rho_used = self._get_rho(league)
        return {
            "home":            round(max(home_win, 0.0), 4),
            "draw":            round(max(draw,     0.0), 4),
            "away":            round(max(away_win, 0.0), 4),
            "over25":          round(max(min(over25, 1.0), 0.0), 4),
            "over35":          round(max(min(over35, 1.0), 0.0), 4),
            "btts":            round(max(min(btts,   1.0), 0.0), 4),
            "xg_home":         round(lam, 2),
            "xg_away":         round(mu,  2),
            "rho":             round(rho_used, 4),
            "r_nb":            round(self.r_nb, 2),
            "correct_scores":  correct_scores,
            "score_grid":      score_grid,
            "score_grid_size": grid.shape[0],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "attack":         self.attack,
            "defence":        self.defence,
            "home_adv":       self.home_adv,
            "rho":            self.rho,
            "rho_by_league":  self.rho_by_league,
            "r_nb":           self.r_nb,
            "teams":          self._teams,
            "fitted":         self._fitted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DixonColesModel":
        dc = cls()
        dc.attack         = {k: float(v) for k, v in data.get("attack",  {}).items()}
        dc.defence        = {k: float(v) for k, v in data.get("defence", {}).items()}
        dc.home_adv       = float(data.get("home_adv", 0.3))
        dc.rho            = float(data.get("rho",      -0.1))
        dc.rho_by_league  = {k: float(v) for k, v in data.get("rho_by_league", {}).items()}
        dc.r_nb           = float(data.get("r_nb", _NB_R_INIT))
        dc._teams         = data.get("teams", [])
        dc._fitted        = data.get("fitted", bool(dc.attack))
        return dc


# ─── I/O ──────────────────────────────────────────────────────────────────────

def save_dc_model(dc: DixonColesModel) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(DC_PARAMS_PATH, "w") as f:
        json.dump(dc.to_dict(), f, indent=2)
    n = len(dc.attack)
    print(f"  [dc] Dixon-Coles saved -> {DC_PARAMS_PATH} ({n} teams, "
          f"r_nb={dc.r_nb:.1f}, {len(dc.rho_by_league)} league rhos)")


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
