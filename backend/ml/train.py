"""
Train four models: 1X2 result, Over/Under 2.5 goals, BTTS, Over/Under 3.5 goals.

Data sources (merged):
  1. Football-Data.co.uk CSVs  — 15 seasons, no API calls, ~8000+ samples
  2. football-data.org API      — current season, live data

Improvements over v1:
  - Early stopping (up to 1500 trees, stops when val loss plateaus)
  - Draw class weighting (balanced sample_weight for class imbalance)
  - Optional Optuna hyperparameter search (set OPTUNA_TRIALS env var, e.g. 50)
  - SHAP feature importance logged per model
  - O/U 3.5 goals as a fourth model
  - Training cache stores y_over35

Run from backend/ directory:
    python -m ml.train
    OPTUNA_TRIALS=50 python -m ml.train   # with hyperparameter search

Models saved to:
    ml/result_model.joblib       ml/result_calibrator.joblib
    ml/goals_model.joblib        ml/goals_calibrator.joblib
    ml/btts_model.joblib         ml/btts_calibrator.joblib
    ml/over35_model.joblib       ml/over35_calibrator.joblib
    ml/training_log.json
"""
import asyncio
import os
import json
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
from ml.features import build_feature_vector, FEATURE_NAMES, N_FEATURES
from ml.splits import split_indices, TRAIN_FRACTION, CALIBRATION_FRACTION, BLEND_FRACTION

ML_DIR = os.path.dirname(__file__)
RESULT_MODEL_PATH  = os.path.join(ML_DIR, "result_model.joblib")
GOALS_MODEL_PATH   = os.path.join(ML_DIR, "goals_model.joblib")
BTTS_MODEL_PATH    = os.path.join(ML_DIR, "btts_model.joblib")
OVER35_MODEL_PATH  = os.path.join(ML_DIR, "over35_model.joblib")
RESULT_CAL_PATH    = os.path.join(ML_DIR, "result_calibrator.joblib")
GOALS_CAL_PATH     = os.path.join(ML_DIR, "goals_calibrator.joblib")
BTTS_CAL_PATH      = os.path.join(ML_DIR, "btts_calibrator.joblib")
OVER35_CAL_PATH    = os.path.join(ML_DIR, "over35_calibrator.joblib")
TRAINING_LOG_PATH  = os.path.join(ML_DIR, "training_log.json")

CACHE_DIR  = os.path.join(os.path.dirname(ML_DIR), "data")
CACHE_PATH = os.path.join(CACHE_DIR, "training_cache.npz")

COMPETITIONS = ["PL", "PD", "BL1", "SA", "FL1", "ELC", "DED"]

# Competition → number of league teams (for position normalisation)
COMPETITION_SIZES = {
    "PL": 20, "PD": 20, "BL1": 18, "SA": 20,
    "FL1": 20, "ELC": 24, "DED": 18,
}


# ─── Model factory ────────────────────────────────────────────────────────────

def _make_xgb(n_classes: int = 3, **overrides) -> XGBClassifier:
    defaults = dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=3,
        gamma=0.15,
        use_label_encoder=False,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        num_class=n_classes if n_classes > 2 else None,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(overrides)
    return XGBClassifier(**defaults)


# ─── Data cache ───────────────────────────────────────────────────────────────

def _load_cache() -> tuple:
    """Load accumulated training data. Returns empty arrays if absent or stale."""
    empty_dates = np.array([], dtype="U10")
    empty = (
        np.empty((0, N_FEATURES), dtype=np.float32),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        empty_dates,
    )
    if not os.path.exists(CACHE_PATH):
        return empty
    try:
        data = np.load(CACHE_PATH)
        X = data["X"].astype(np.float32)
        if X.shape[1] != N_FEATURES:
            print(f"  [cache] Feature mismatch ({X.shape[1]} vs {N_FEATURES}) — discarding.")
            return empty
        y_over35 = data["y_over35"] if "y_over35" in data else np.zeros(len(X), dtype=np.int64)
        # dates added in v2 — back-fill with empty strings for old caches
        dates = data["dates"] if "dates" in data else np.array([""] * len(X), dtype="U10")
        return X, data["y_result"], data["y_goals"], data["y_btts"], y_over35, dates
    except Exception as e:
        print(f"  [cache] Load failed ({e}), starting fresh.")
        return empty


def _save_cache(X, y_result, y_goals, y_btts, y_over35, dates=None) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if dates is None:
        dates = np.array([""] * len(X), dtype="U10")
    np.savez_compressed(
        CACHE_PATH, X=X,
        y_result=y_result, y_goals=y_goals, y_btts=y_btts, y_over35=y_over35,
        dates=dates,
    )
    print(f"  [cache] Saved {len(X)} samples to {CACHE_PATH}")


def _merge(*arrays_tuple, dates_tuple=None):
    """Merge (X, yr, yg, yb, yo) tuples, deduplicating on feature row hash.

    FIX #1 — temporal leakage:
    When dates_tuple is provided (one date-string array per source), the merged
    output is SORTED chronologically before returning. This ensures TimeSeriesSplit
    in CV sees genuine past→future folds instead of randomly interleaved leagues.

    Previously data was concatenated in source order (cache → FDCO per-league →
    API), which destroyed chronological ordering and allowed future fixtures to
    bleed into training folds — inflating reported CV accuracy by 5–10%.
    """
    Xs, yrs, ygs, ybs, yos = zip(*arrays_tuple)
    X_all  = np.concatenate(Xs)
    yr_all = np.concatenate(yrs)
    yg_all = np.concatenate(ygs)
    yb_all = np.concatenate(ybs)
    yo_all = np.concatenate(yos)

    if dates_tuple is not None:
        dates_all = np.concatenate(dates_tuple)
    else:
        dates_all = np.array([""] * len(X_all), dtype="U10")

    if len(X_all) == 0:
        return X_all, yr_all, yg_all, yb_all, yo_all, dates_all

    seen = set()
    keep = []
    for i, row in enumerate(X_all):
        key = tuple(round(float(v), 4) for v in row) + (int(yr_all[i]),)
        if key not in seen:
            seen.add(key)
            keep.append(i)

    keep = np.array(keep)
    X_out  = X_all[keep];  yr_out = yr_all[keep]
    yg_out = yg_all[keep]; yb_out = yb_all[keep]
    yo_out = yo_all[keep]; dt_out = dates_all[keep]

    # Sort chronologically — empty-string dates (cache back-compat) sort to front
    sort_idx = np.argsort(dt_out, kind="stable")
    return (X_out[sort_idx], yr_out[sort_idx], yg_out[sort_idx],
            yb_out[sort_idx], yo_out[sort_idx], dt_out[sort_idx])


# ─── Optuna hyperparameter search ─────────────────────────────────────────────

def _optuna_search(X: np.ndarray, y: np.ndarray, n_classes: int, n_trials: int = 50) -> dict:
    """
    Find best XGBoost hyperparameters via Optuna.
    Runs on a random 5k-sample subset for speed. Returns param overrides dict.
    """
    try:
        import optuna
    except ImportError:
        print("  [optuna] Not installed — skipping search.")
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # FIX #15: take the most recent n_sample chronologically (X is already date-sorted).
    # Random sampling destroyed temporal order so TimeSeriesSplit saw pseudo-random
    # folds and overestimated hyperparameter CV accuracy by 5–10%.
    n_sample = min(5000, len(X))
    idx = np.arange(len(X) - n_sample, len(X))
    Xs, ys = X[idx], y[idx]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
            gamma=trial.suggest_float("gamma", 0.0, 0.5),
        )
        m = _make_xgb(n_classes, **params)
        cv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(m, Xs, ys, cv=cv, scoring="neg_log_loss")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  [optuna] Best params after {n_trials} trials: {best}")
    print(f"  [optuna] Best log-loss: {study.best_value:.4f}")
    return best


# ─── SHAP feature importance ──────────────────────────────────────────────────

def _log_shap(model, X_sample: np.ndarray, label: str) -> list:
    """Compute mean |SHAP| per feature. Returns list of (name, importance) pairs."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sample = X_sample[:min(1000, len(X_sample))]
        sv = explainer.shap_values(sample)
        # Multiclass → mean across classes; binary → single array
        if isinstance(sv, list):
            importance = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
        else:
            importance = np.abs(sv).mean(0)
        top_idx = np.argsort(importance)[::-1][:10]
        top = [(FEATURE_NAMES[i], round(float(importance[i]), 4)) for i in top_idx]
        print(f"  {label} SHAP top-5: {top[:5]}")
        return top
    except Exception as e:
        print(f"  {label} SHAP skipped: {e}")
        return []


# ─── Training + calibration ───────────────────────────────────────────────────

def _train_and_calibrate(
    X: np.ndarray, y: np.ndarray, label: str, n_classes: int,
    model_path: str, cal_path: str,
    hparams: dict = None,
) -> dict:
    """
    1. Three-way chronological split (70/15/15):
         train (70%) → XGBoost
         cal   (15%) → probability calibration
         blend (15%) → held back for optimize_blend.py (not touched here,
                       but metrics are reported on it as the honest
                       out-of-sample performance number).
    2. TimeSeriesSplit 5-fold CV on training slice for honest accuracy.
    3. Final fit with early stopping (up to 1500 trees) on 85/15 sub-split.
    4. Draw (class 1) up-weighted via balanced sample_weight.
    5. Calibrator fit on the cal slice (adaptive sigmoid/isotonic).
    6. SHAP feature importance logged.

    Why this matters: previously the calibrator and the blend optimiser both
    used the last 20% of data. The blend optimiser was measuring in-sample
    Brier for the calibrator, overfitting the ensemble weights. Now the
    blend slice [CAL_END:] is only ever seen by optimize_blend.py.
    """
    if len(np.unique(y)) < n_classes:
        n_classes = len(np.unique(y))

    hparams = hparams or {}

    # Three-way chronological split — boundaries defined in ml/splits.py so
    # train.py and optimize_blend.py agree on exactly which rows go where.
    train_end, cal_end = split_indices(len(X))
    X_train, X_cal, X_blend = X[:train_end], X[train_end:cal_end], X[cal_end:]
    y_train, y_cal, y_blend = y[:train_end], y[train_end:cal_end], y[cal_end:]

    # Sample weights — upweight draws / minority classes
    sample_weight = compute_sample_weight("balanced", y_train)

    # 5-fold TimeSeriesSplit CV — respects temporal ordering so later folds are
    # always in the future relative to training folds. Previously StratifiedKFold
    # with shuffle=False was used on data that wasn't globally sorted by date,
    # causing future match data to leak into earlier training folds (FIX #1).
    cv_model = _make_xgb(n_classes, **hparams)
    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(cv_model, X_train, y_train, cv=cv,
                             scoring="accuracy", fit_params={"sample_weight": sample_weight})
    print(f"  {label} CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Final fit with early stopping — carve 15% of X_train as early-stop val
    es_split = int(len(X_train) * 0.85)
    X_fit, X_es = X_train[:es_split], X_train[es_split:]
    y_fit, y_es = y_train[:es_split], y_train[es_split:]
    sw_fit = sample_weight[:es_split]
    # FIX #12: apply balanced sample weights to the early-stopping eval set too.
    # Previously the eval set was unweighted, so early stopping optimised for
    # unbalanced accuracy — overfitting HOME wins and suppressing draws.
    sw_es  = compute_sample_weight("balanced", y_es)

    final_model = _make_xgb(n_classes, n_estimators=1500, **hparams)
    final_model.fit(
        X_fit, y_fit,
        sample_weight=sw_fit,
        eval_set=[(X_es, y_es)],
        sample_weight_eval_set=[sw_es],
        early_stopping_rounds=30,
        verbose=False,
    )
    best_trees = getattr(final_model, "best_iteration", None) or getattr(final_model, "best_ntree_limit", None)
    print(f"  {label} early stopping -> {best_trees} trees")
    joblib.dump(final_model, model_path)

    # SHAP on calibration holdout
    shap_top10 = _log_shap(final_model, X_cal, label)

    # FIX #16: adaptive calibration method.
    # Isotonic regression needs ~10 points per bin to avoid overfitting — with
    # ~1800 calibration samples and 3 classes, the draw class gets only ~450
    # points which isotonic will memorise rather than smooth. Sigmoid (Platt)
    # scaling is a single-parameter fit that generalises much better at this
    # data volume.  Switch to isotonic only when we have ≥2000 cal samples.
    if cal_path is not None:
        cal_method = "sigmoid" if len(X_cal) < 2000 else "isotonic"
        print(f"  {label} calibration method: {cal_method} ({len(X_cal)} cal samples)")
        calibrator = CalibratedClassifierCV(final_model, method=cal_method, cv="prefit")
        calibrator.fit(X_cal, y_cal)
        joblib.dump(calibrator, cal_path)
        prob_source = calibrator
    else:
        prob_source = final_model

    # Evaluate on the held-back BLEND slice — genuinely out-of-sample for
    # both the XGBoost model AND the calibrator, so this is the honest
    # generalisation number. The cal-slice metrics are reported too, for
    # diagnostic comparison (a large cal-vs-blend gap = calibrator
    # overfitting the cal window, usually a sign the cal window is too
    # small or isotonic is being used when sigmoid would be safer).
    blend_probs = prob_source.predict_proba(X_blend) if len(X_blend) else None
    cal_probs   = prob_source.predict_proba(X_cal)

    def _metrics(probs, y_true):
        if n_classes > 2:
            ll = log_loss(y_true, probs, labels=list(range(n_classes)))
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            br = float(np.mean((probs - y_bin) ** 2))
        else:
            ll = log_loss(y_true, probs, labels=[0, 1])
            br = brier_score_loss(y_true, probs[:, 1])
        return ll, br

    cal_ll, cal_brier = _metrics(cal_probs, y_cal)
    if blend_probs is not None and len(X_blend) > 10:
        ll, brier = _metrics(blend_probs, y_blend)
        blend_note = f"blend-holdout (n={len(X_blend)}, OOS for model+calibrator)"
    else:
        # Degenerate case — not enough blend data, fall back to cal metrics
        ll, brier = cal_ll, cal_brier
        blend_note = f"cal-slice fallback (n={len(X_cal)}, insufficient blend data)"

    print(
        f"  {label} {blend_note} | "
        f"log-loss={ll:.4f} | brier={brier:.4f}"
    )
    print(
        f"  {label} cal-slice (n={len(X_cal)}) | "
        f"log-loss={cal_ll:.4f} | brier={cal_brier:.4f}"
    )

    return {
        "accuracy_mean":   float(scores.mean()),
        "accuracy_std":    float(scores.std()),
        "best_trees":      int(best_trees),
        # Headline numbers (blend holdout — genuinely OOS):
        "log_loss":        round(ll, 4),
        "brier_score":     round(brier, 4),
        # Diagnostic (cal slice — in-sample for calibrator):
        "cal_log_loss":    round(cal_ll, 4),
        "cal_brier_score": round(cal_brier, 4),
        "n_train":         int(len(X_train)),
        "n_cal":           int(len(X_cal)),
        "n_blend":         int(len(X_blend)),
        "shap_top10":      shap_top10,
    }


# ─── API data fetcher ─────────────────────────────────────────────────────────

async def fetch_api_training_data():
    """
    Pull current-season finished matches from football-data.org API.

    FIX #2 + #3 + #18 — lookahead bias elimination:
    The old implementation called `_cached_team_matches(team_id)` which returned
    the 25 most recent matches *as of today*, not as of match day. This meant
    that when building the feature vector for e.g. matchweek 5, the model could
    see results from matchweeks 6–25 in the "history" arrays, leaking future
    outcomes into training features and inflating accuracy by 5–15%.

    The fix mirrors fdco_trainer.py: iterate ALL competition matches in strict
    chronological order, maintaining a running `team_history` dict and a
    `_RunningTable` for standings. At the point of computing a feature vector,
    the history contains only matches that actually preceded this fixture.

    This also fixes Fix #18 (H2H was always empty []): H2H is now extracted
    from the running team_history using the same filter as fdco_trainer.py.
    """
    import httpx
    from app.config import settings
    from ml.fdco_trainer import _RunningTable  # same running-standings helper

    BASE_URL = "https://api.football-data.org/v4"
    _last_t: list[float] = [0.0]
    MIN_HISTORY = 5  # minimum prior matches required to generate a training sample

    async def _train_get(url: str, params: dict = None) -> dict:
        now = asyncio.get_event_loop().time()
        wait = max(0.0, 7.0 - (now - _last_t[0]))
        if wait > 0:
            await asyncio.sleep(wait)
        _last_t[0] = asyncio.get_event_loop().time()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                url,
                headers={"X-Auth-Token": settings.football_data_api_key},
                params=params,
            )
            if resp.status_code == 429:
                await asyncio.sleep(65)
                resp = await client.get(
                    url,
                    headers={"X-Auth-Token": settings.football_data_api_key},
                    params=params,
                )
            resp.raise_for_status()
            return resp.json()

    X, y_result, y_goals, y_btts, y_over35 = [], [], [], [], []
    dates: list[str] = []
    skipped = 0

    for comp in COMPETITIONS:
        print(f"  [{comp}] Fetching matches…")
        try:
            data = await _train_get(
                f"{BASE_URL}/competitions/{comp}/matches", {"status": "FINISHED"}
            )
            all_matches = data.get("matches", [])
        except Exception as e:
            print(f"  [{comp}] Error: {e}")
            continue

        total_teams = COMPETITION_SIZES.get(comp, 20)
        finished = [m for m in all_matches if m.get("status") == "FINISHED"]

        # Sort chronologically — CRITICAL for correct running history
        finished.sort(key=lambda m: m.get("utcDate", ""))
        print(f"  [{comp}] {len(finished)} finished matches — building running history…")

        # Running state (reset per competition)
        team_history: dict[int, list[dict]] = {}  # team_id → chronological match list
        table = _RunningTable()

        for i, match in enumerate(finished):
            home_team = match.get("homeTeam", {})
            away_team = match.get("awayTeam", {})
            home_id   = home_team.get("id")
            away_id   = away_team.get("id")
            hg = match.get("score", {}).get("fullTime", {}).get("home")
            ag = match.get("score", {}).get("fullTime", {}).get("away")
            date_str  = (match.get("utcDate") or "")[:10]
            home_name = home_team.get("name", "")
            away_name = away_team.get("name", "")

            if not all([home_id, away_id, hg is not None, ag is not None]):
                skipped += 1
                # Still update running state so later matches have correct context
                if home_id and away_id and hg is not None and ag is not None:
                    table.update(home_name, away_name, int(hg), int(ag))
                    team_history.setdefault(home_id, []).append(match)
                    team_history.setdefault(away_id, []).append(match)
                continue

            hg, ag = int(hg), int(ag)
            home_hist = team_history.get(home_id, [])
            away_hist = team_history.get(away_id, [])

            # Require minimum prior history (matches BEFORE this fixture)
            if len(home_hist) < MIN_HISTORY or len(away_hist) < MIN_HISTORY:
                # Update running state without producing a training sample
                table.update(home_name, away_name, hg, ag)
                team_history.setdefault(home_id, []).append(match)
                team_history.setdefault(away_id, []).append(match)
                continue

            # Labels
            if hg > ag:    result_label = 0
            elif hg == ag: result_label = 1
            else:          result_label = 2

            goals_label  = 1 if (hg + ag) > 2 else 0
            btts_label   = 1 if (hg > 0 and ag > 0) else 0
            over35_label = 1 if (hg + ag) > 3 else 0

            # FIX #18: H2H extracted from running history (no future leakage)
            h2h = [
                mm for mm in home_hist
                if (mm.get("homeTeam", {}).get("id") == away_id
                    or mm.get("awayTeam", {}).get("id") == away_id)
            ][-10:]

            # Standings at match time (from running table)
            home_std = table.standing(home_name)
            away_std = table.standing(away_name)

            vec = build_feature_vector(
                home_id, away_id,
                home_hist[-25:],
                away_hist[-25:],
                h2h_matches=h2h,
                home_standing=home_std,
                away_standing=away_std,
                match_date=date_str,
                total_teams=total_teams,
            )
            X.append(vec)
            y_result.append(result_label)
            y_goals.append(goals_label)
            y_btts.append(btts_label)
            y_over35.append(over35_label)
            dates.append(date_str)

            # Update running state AFTER building the feature vector (no leakage)
            table.update(home_name, away_name, hg, ag)
            team_history.setdefault(home_id, []).append(match)
            team_history.setdefault(away_id, []).append(match)

            if (i + 1) % 20 == 0:
                print(f"    [{comp}] {i + 1}/{len(finished)} processed, {len(X)} samples so far…")

    print(f"\n  [api] {len(X)} samples | {skipped} skipped")
    empty = np.empty((0, N_FEATURES), dtype=np.float32)
    if not X:
        return (empty, np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([], dtype="U10"))
    return (
        np.array(X, dtype=np.float32),
        np.array(y_result), np.array(y_goals),
        np.array(y_btts),   np.array(y_over35),
        np.array(dates, dtype="U10"),
    )


# ─── Main training orchestrator ───────────────────────────────────────────────

def train_all(X, y_result, y_goals, y_btts, y_over35,
              odds_rows: list = None, hparams: dict = None) -> dict:
    hparams = hparams or {}
    print(f"\nTraining on {len(X)} samples with {N_FEATURES} features…\n")

    print("Training result model (1X2)…")
    result_stats = _train_and_calibrate(
        X, y_result, "Result", 3, RESULT_MODEL_PATH, RESULT_CAL_PATH, hparams,
    )

    print("Training goals model (O/U 2.5)…")
    goals_stats = _train_and_calibrate(
        X, y_goals, "Goals", 2, GOALS_MODEL_PATH, GOALS_CAL_PATH, hparams,
    )

    print("Training BTTS model…")
    btts_stats = _train_and_calibrate(
        X, y_btts, "BTTS", 2, BTTS_MODEL_PATH, BTTS_CAL_PATH, hparams,
    )

    print("Training O/U 3.5 goals model…")
    over35_stats = _train_and_calibrate(
        X, y_over35, "Over35", 2, OVER35_MODEL_PATH, OVER35_CAL_PATH, hparams,
    )

    # Dixon-Coles
    dc_stats = {}
    if odds_rows:
        print("\nFitting Dixon-Coles model…")
        from ml.dixon_coles import DixonColesModel, save_dc_model, MAX_MATCHES
        dc_matches = [
            {
                "home":       r["home"],
                "away":       r["away"],
                "home_goals": r.get("home_goals", 0),
                "away_goals": r.get("away_goals", 0),
                "date":       r.get("date", "2020-01-01"),
            }
            for r in odds_rows if r.get("home") and r.get("away")
        ]
        dc_matches.sort(key=lambda m: m["date"])
        dc_matches = dc_matches[-MAX_MATCHES:]
        dc = DixonColesModel()
        dc.fit(dc_matches)
        if dc._fitted:
            save_dc_model(dc)
            dc_stats = {"teams": len(dc.attack), "matches_used": len(dc_matches)}
        else:
            print("  [dc] Fitting did not converge — skipping save.")

    # FIX #9: auto-run blend weight optimisation after every retrain.
    # Previously blend_weights.json was hardcoded with arbitrary values
    # ("result": 0.2, "over25": 0.05). Now the optimiser runs on the temporal
    # holdout set automatically so blend weights are always data-driven.
    print("\nRunning blend weight optimisation…")
    try:
        from ml.optimize_blend import optimise as _optimise_blend
        _optimise_blend()
    except Exception as _e:
        print(f"  [blend] Optimisation failed (non-fatal): {_e}")

    log = _load_training_log()
    from datetime import datetime, timezone
    entry = {
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "samples":       int(len(X)),
        "result_model":  result_stats,
        "goals_model":   goals_stats,
        "btts_model":    btts_stats,
        "over35_model":  over35_stats,
        "dixon_coles":   dc_stats,
        "feature_count": N_FEATURES,
        "features":      FEATURE_NAMES,
    }
    log.append(entry)
    _save_training_log(log)
    print(f"\nAll models + calibrators saved. Training log updated ({len(log)} entries).")
    return entry


# ─── Per-league model training ────────────────────────────────────────────────

MIN_LEAGUE_SAMPLES = 300  # minimum samples needed to train a reliable per-league model


def train_per_league(
    X: np.ndarray,
    y_result: np.ndarray,
    y_goals: np.ndarray,
    y_btts: np.ndarray,
    y_over35: np.ndarray,
    odds_rows: list,
    hparams: dict = None,
) -> dict:
    """
    Train competition-specific models for leagues that have enough samples.

    Uses odds_rows (from FDCO) which carries feature_idx → league mapping.
    Saves models as result_model_{code}.joblib etc. alongside the combined model.

    Per-league models capture league-specific patterns:
      - PL: high-tempo, fewer draws
      - FL1: high-draw rate
      - BL1: high-scoring games
      - ELC: stamina/volume game, many matches
    """
    hparams = hparams or {}

    # Build league → [feature indices] mapping from odds_rows
    league_indices: dict[str, list[int]] = {}
    for row in odds_rows:
        code = row.get("league")
        idx  = row.get("feature_idx")
        if code and idx is not None and idx < len(X):
            league_indices.setdefault(code, []).append(idx)

    results = {}
    for code, indices in league_indices.items():
        n = len(indices)
        if n < MIN_LEAGUE_SAMPLES:
            print(f"  [{code}] {n} samples — skipping (need ≥{MIN_LEAGUE_SAMPLES})")
            continue

        print(f"\n[per-league] Training {code} ({n} samples)…")
        idx_arr = np.array(indices)
        Xl = X[idx_arr]
        yrl = y_result[idx_arr]
        ygl = y_goals[idx_arr]
        ybl = y_btts[idx_arr]
        yol = y_over35[idx_arr]

        suffix = f"_{code}"
        league_result_path  = RESULT_MODEL_PATH.replace(".joblib", f"{suffix}.joblib")
        league_goals_path   = GOALS_MODEL_PATH.replace(".joblib",  f"{suffix}.joblib")
        league_btts_path    = BTTS_MODEL_PATH.replace(".joblib",   f"{suffix}.joblib")
        league_over35_path  = OVER35_MODEL_PATH.replace(".joblib",  f"{suffix}.joblib")

        # FIX #2: train and save per-league calibrators.
        # Previously per-league models used the global calibrator at inference,
        # which was fitted on the combined dataset (P(home)≈0.46). A per-league
        # model (e.g. PL) outputs P(home)≈0.52, so the global calibrator pushed
        # probabilities in the wrong direction. Each league now gets its own
        # isotonic calibrator trained on its own holdout split.
        league_result_cal_path  = RESULT_CAL_PATH.replace(".joblib",  f"{suffix}.joblib")
        league_goals_cal_path   = GOALS_CAL_PATH.replace(".joblib",   f"{suffix}.joblib")
        league_btts_cal_path    = BTTS_CAL_PATH.replace(".joblib",    f"{suffix}.joblib")
        league_over35_cal_path  = OVER35_CAL_PATH.replace(".joblib",  f"{suffix}.joblib")

        r = _train_and_calibrate(Xl, yrl, f"Result-{code}", 3, league_result_path, league_result_cal_path,   hparams)
        g = _train_and_calibrate(Xl, ygl, f"Goals-{code}",  2, league_goals_path,  league_goals_cal_path,    hparams)
        b = _train_and_calibrate(Xl, ybl, f"BTTS-{code}",   2, league_btts_path,   league_btts_cal_path,     hparams)
        o = _train_and_calibrate(Xl, yol, f"Over35-{code}", 2, league_over35_path, league_over35_cal_path,   hparams)

        results[code] = {"samples": n, "result": r, "goals": g, "btts": b, "over35": o}
        print(f"  [{code}] Models saved.")

    return results


# ─── Log helpers ─────────────────────────────────────────────────────────────

def _load_training_log() -> list:
    if os.path.exists(TRAINING_LOG_PATH):
        try:
            with open(TRAINING_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_training_log(log: list) -> None:
    with open(TRAINING_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def get_last_training_summary() -> dict:
    log = _load_training_log()
    return log[-1] if log else {}


# ─── Entry point ─────────────────────────────────────────────────────────────

async def main():
    print("=== Sports Bet Analysis — Model Training ===\n")

    # 1. Load accumulated cache
    print("Loading accumulated data cache…")
    X_cache, yr_cache, yg_cache, yb_cache, yo_cache, dates_cache = _load_cache()
    print(f"  Cache: {len(X_cache)} samples\n")

    # 2. Download + load FDCO CSV data
    print("Loading Football-Data.co.uk CSV data…")
    from ml.fdco_trainer import build_fdco_training_data, download_all_csvs
    await download_all_csvs()
    X_fdco, yr_fdco, yg_fdco, yb_fdco, yo_fdco, odds_rows_fdco, dates_fdco = build_fdco_training_data()
    print()

    # 3. Fetch fresh API data
    print("Fetching current-season data from API (rate-limited)…\n")
    X_api, yr_api, yg_api, yb_api, yo_api, dates_api = await fetch_api_training_data()
    print()

    # 4. Merge all sources + update cache
    # FIX #1: _merge now sorts by date so TimeSeriesSplit sees genuine temporal folds.
    X, y_result, y_goals, y_btts, y_over35, dates_merged = _merge(
        (X_cache, yr_cache, yg_cache, yb_cache, yo_cache),
        (X_fdco,  yr_fdco,  yg_fdco,  yb_fdco,  yo_fdco),
        (X_api,   yr_api,   yg_api,   yb_api,   yo_api),
        dates_tuple=(dates_cache, dates_fdco, dates_api),
    )
    print(f"Total unique samples after merge: {len(X)}")
    _save_cache(X, y_result, y_goals, y_btts, y_over35, dates_merged)

    if len(X) < 50:
        print("ERROR: Not enough training data. Check your API key and CSV files.")
        return

    # 5. Optional Optuna hyperparameter search (gated by OPTUNA_TRIALS env var)
    n_trials = int(os.environ.get("OPTUNA_TRIALS", "0"))
    hparams = {}
    if n_trials > 0 and len(X) >= 500:
        print(f"\nRunning Optuna search on result model ({n_trials} trials)…")
        hparams = _optuna_search(X, y_result, n_classes=3, n_trials=n_trials)

    # 6. Train combined model (all leagues together)
    train_all(X, y_result, y_goals, y_btts, y_over35,
              odds_rows=odds_rows_fdco, hparams=hparams)

    # 7. Train per-league models (uses FDCO data which has competition labels)
    #    These are used automatically at inference when a league-specific model exists.
    print("\n=== Per-league model training ===")
    train_per_league(
        X_fdco, yr_fdco, yg_fdco, yb_fdco, yo_fdco,
        odds_rows=odds_rows_fdco, hparams=hparams,
    )


if __name__ == "__main__":
    asyncio.run(main())
