"""
Train three models: 1X2 result, Over/Under 2.5 goals, BTTS.

Data sources (merged):
  1. Football-Data.co.uk CSVs  — 3-5 seasons, no API calls, ~3000+ samples
  2. football-data.org API      — current season, live data

After training, each XGBoost model is isotonic-calibrated on a 20% holdout
to produce well-calibrated probabilities. Calibrators are saved alongside models.

Accumulated feature cache (data/training_cache.npz) is loaded at startup and
merged with freshly fetched data, so the dataset grows with every retrain.

Run from backend/ directory:
    python -m ml.train

Models saved to:
    ml/result_model.joblib       ml/result_calibrator.joblib
    ml/goals_model.joblib        ml/goals_calibrator.joblib
    ml/btts_model.joblib         ml/btts_calibrator.joblib
    ml/training_log.json
"""
import asyncio
import os
import json
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from ml.features import build_feature_vector, FEATURE_NAMES, N_FEATURES

ML_DIR = os.path.dirname(__file__)
RESULT_MODEL_PATH  = os.path.join(ML_DIR, "result_model.joblib")
GOALS_MODEL_PATH   = os.path.join(ML_DIR, "goals_model.joblib")
BTTS_MODEL_PATH    = os.path.join(ML_DIR, "btts_model.joblib")
RESULT_CAL_PATH    = os.path.join(ML_DIR, "result_calibrator.joblib")
GOALS_CAL_PATH     = os.path.join(ML_DIR, "goals_calibrator.joblib")
BTTS_CAL_PATH      = os.path.join(ML_DIR, "btts_calibrator.joblib")
TRAINING_LOG_PATH  = os.path.join(ML_DIR, "training_log.json")

# Accumulation cache — grows with every retrain
CACHE_DIR  = os.path.join(os.path.dirname(ML_DIR), "data")
CACHE_PATH = os.path.join(CACHE_DIR, "training_cache.npz")

COMPETITIONS = ["PL", "PD", "BL1", "SA", "FL1", "ELC", "DED"]


# ─── Model factory ────────────────────────────────────────────────────────────

def _make_xgb(n_classes: int = 3) -> XGBClassifier:
    return XGBClassifier(
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


# ─── Data cache ───────────────────────────────────────────────────────────────

def _load_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load accumulated training data from disk. Returns empty arrays if none."""
    if not os.path.exists(CACHE_PATH):
        return (
            np.empty((0, N_FEATURES), dtype=np.float32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    try:
        data = np.load(CACHE_PATH)
        X = data["X"].astype(np.float32)
        # Handle feature dimension mismatch (model upgrade added features)
        if X.shape[1] != N_FEATURES:
            print(f"  [cache] Feature mismatch ({X.shape[1]} vs {N_FEATURES}) — discarding old cache.")
            return (
                np.empty((0, N_FEATURES), dtype=np.float32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        return X, data["y_result"], data["y_goals"], data["y_btts"]
    except Exception as e:
        print(f"  [cache] Load failed ({e}), starting fresh.")
        return (
            np.empty((0, N_FEATURES), dtype=np.float32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )


def _save_cache(X, y_result, y_goals, y_btts) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(CACHE_PATH, X=X, y_result=y_result, y_goals=y_goals, y_btts=y_btts)
    print(f"  [cache] Saved {len(X)} samples to {CACHE_PATH}")


def _merge(*arrays_tuple):
    """Merge (X, yr, yg, yb) tuples, deduplicating on feature row hash."""
    Xs, yrs, ygs, ybs = zip(*arrays_tuple)
    X_all  = np.concatenate(Xs)
    yr_all = np.concatenate(yrs)
    yg_all = np.concatenate(ygs)
    yb_all = np.concatenate(ybs)

    if len(X_all) == 0:
        return X_all, yr_all, yg_all, yb_all

    # Deduplicate by rounding features to 4dp and hashing
    seen = set()
    keep = []
    for i, row in enumerate(X_all):
        key = tuple(round(float(v), 4) for v in row) + (int(yr_all[i]),)
        if key not in seen:
            seen.add(key)
            keep.append(i)

    return X_all[keep], yr_all[keep], yg_all[keep], yb_all[keep]


# ─── API data fetcher ─────────────────────────────────────────────────────────

async def fetch_api_training_data():
    """Pull current-season finished matches from football-data.org API."""
    from app.services.football_api import get_finished_matches, get_team_matches, get_standings

    X, y_result, y_goals, y_btts = [], [], [], []
    skipped = 0
    team_cache: dict[int, list] = {}

    async def _cached_team_matches(team_id: int) -> list:
        if team_id not in team_cache:
            await asyncio.sleep(7)
            try:
                team_cache[team_id] = await get_team_matches(team_id, limit=25)
            except Exception:
                team_cache[team_id] = []
        return team_cache[team_id]

    for comp_idx, comp in enumerate(COMPETITIONS):
        if comp_idx > 0:
            await asyncio.sleep(8)

        print(f"  [{comp}] Fetching matches…")
        try:
            matches = await get_finished_matches(comp, limit=150)
            await asyncio.sleep(8)
            standings_table = await get_standings(comp)
            standing_map = {row["team"]["id"]: row for row in standings_table}
        except Exception as e:
            print(f"  [{comp}] Error: {e}")
            continue

        finished = [m for m in matches if m.get("status") == "FINISHED"]
        print(f"  [{comp}] {len(finished)} finished matches — fetching team histories…")

        for i, match in enumerate(finished):
            home_id = match.get("homeTeam", {}).get("id")
            away_id = match.get("awayTeam", {}).get("id")
            hg = match.get("score", {}).get("fullTime", {}).get("home")
            ag = match.get("score", {}).get("fullTime", {}).get("away")
            date_str = (match.get("utcDate") or "")[:10]

            if not all([home_id, away_id, hg is not None, ag is not None]):
                skipped += 1
                continue

            if hg > ag:    result_label = 0
            elif hg == ag: result_label = 1
            else:          result_label = 2

            goals_label = 1 if (hg + ag) > 2 else 0
            btts_label  = 1 if (hg > 0 and ag > 0) else 0

            home_m = await _cached_team_matches(home_id)
            away_m = await _cached_team_matches(away_id)

            vec = build_feature_vector(
                home_id, away_id,
                home_m, away_m,
                h2h_matches=[],
                home_standing=standing_map.get(home_id),
                away_standing=standing_map.get(away_id),
                match_date=date_str,
            )
            X.append(vec)
            y_result.append(result_label)
            y_goals.append(goals_label)
            y_btts.append(btts_label)

            if (i + 1) % 20 == 0:
                print(f"    [{comp}] {i + 1}/{len(finished)} processed, {len(X)} samples so far…")

    print(f"\n  [api] {len(X)} samples | {skipped} skipped | {len(team_cache)} unique teams")
    return (
        np.array(X, dtype=np.float32) if X else np.empty((0, N_FEATURES), dtype=np.float32),
        np.array(y_result), np.array(y_goals), np.array(y_btts),
    )


# ─── Training + calibration ───────────────────────────────────────────────────

def _train_and_calibrate(
    X: np.ndarray, y: np.ndarray, label: str, n_classes: int,
    model_path: str, cal_path: str,
) -> dict:
    """
    1. Temporal 80/20 split (first 80% = train, last 20% = calibration).
       This avoids future-leakage into the calibration set.
    2. 5-fold stratified CV on training split for honest accuracy estimate.
    3. Fit XGBoost on full training split.
    4. Fit isotonic calibrator on temporal holdout.
    5. Save both. Report log-loss + Brier on calibration split.
    """
    if len(np.unique(y)) < n_classes:
        n_classes = len(np.unique(y))

    # Temporal split: first 80% for training, last 20% for calibration
    split = int(len(X) * 0.8)
    X_train, X_cal = X[:split], X[split:]
    y_train, y_cal = y[:split], y[split:]

    model = _make_xgb(n_classes)
    cv = StratifiedKFold(n_splits=5, shuffle=False)   # no shuffle = respects time order
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  {label} CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    # Isotonic calibration on temporal holdout
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_cal, y_cal)
    joblib.dump(calibrator, cal_path)

    # Quality metrics on calibration set
    cal_probs = calibrator.predict_proba(X_cal)
    if n_classes > 2:
        ll = log_loss(y_cal, cal_probs)
        # Multiclass Brier: mean over all classes
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_cal, classes=list(range(n_classes)))
        brier = float(np.mean((cal_probs - y_bin) ** 2))
    else:
        ll    = log_loss(y_cal, cal_probs)
        brier = brier_score_loss(y_cal, cal_probs[:, 1])

    print(
        f"  {label} calibrator — {len(X_cal)} samples | "
        f"log-loss={ll:.4f} | brier={brier:.4f}"
    )

    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std":  float(scores.std()),
        "log_loss":      round(ll, 4),
        "brier_score":   round(brier, 4),
    }


def train_all(X, y_result, y_goals, y_btts, odds_rows: list = None) -> dict:
    print(f"\nTraining on {len(X)} samples with {N_FEATURES} features…\n")

    print("Training result model (1X2)…")
    result_stats = _train_and_calibrate(
        X, y_result, "Result", 3, RESULT_MODEL_PATH, RESULT_CAL_PATH,
    )

    print("Training goals model (O/U 2.5)…")
    goals_stats = _train_and_calibrate(
        X, y_goals, "Goals", 2, GOALS_MODEL_PATH, GOALS_CAL_PATH,
    )

    print("Training BTTS model…")
    btts_stats = _train_and_calibrate(
        X, y_btts, "BTTS", 2, BTTS_MODEL_PATH, BTTS_CAL_PATH,
    )

    # ── Dixon-Coles model ────────────────────────────────────────────────────
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
            for r in odds_rows
            if r.get("home") and r.get("away")
        ]
        # Use most recent matches for DC (time decay handles recency but cap for speed)
        dc_matches.sort(key=lambda m: m["date"])
        dc_matches = dc_matches[-MAX_MATCHES:]
        dc = DixonColesModel()
        dc.fit(dc_matches)
        if dc._fitted:
            save_dc_model(dc)
            dc_stats = {"teams": len(dc.attack), "matches_used": len(dc_matches)}
        else:
            print("  [dc] Fitting did not converge — skipping save.")

    log = _load_training_log()
    from datetime import datetime, timezone
    entry = {
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "samples":       int(len(X)),
        "result_model":  result_stats,
        "goals_model":   goals_stats,
        "btts_model":    btts_stats,
        "dixon_coles":   dc_stats,
        "feature_count": N_FEATURES,
        "features":      FEATURE_NAMES,
    }
    log.append(entry)
    _save_training_log(log)
    print(f"\nAll models + calibrators saved. Training log updated ({len(log)} entries).")
    return entry


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
    X_cache, yr_cache, yg_cache, yb_cache = _load_cache()
    print(f"  Cache: {len(X_cache)} samples\n")

    # 2. Download + load FDCO CSV data (no API calls)
    print("Loading Football-Data.co.uk CSV data…")
    from ml.fdco_trainer import build_fdco_training_data, download_all_csvs
    await download_all_csvs()
    X_fdco, yr_fdco, yg_fdco, yb_fdco, odds_rows_fdco = build_fdco_training_data()
    print()

    # 3. Fetch fresh API data
    print("Fetching current-season data from API (rate-limited)…\n")
    X_api, yr_api, yg_api, yb_api = await fetch_api_training_data()
    print()

    # 4. Merge all sources + update cache
    X, y_result, y_goals, y_btts = _merge(
        (X_cache, yr_cache, yg_cache, yb_cache),
        (X_fdco,  yr_fdco,  yg_fdco,  yb_fdco),
        (X_api,   yr_api,   yg_api,   yb_api),
    )
    print(f"Total unique samples after merge: {len(X)}")
    _save_cache(X, y_result, y_goals, y_btts)

    if len(X) < 50:
        print("ERROR: Not enough training data. Check your API key and CSV files.")
        return

    # Pass odds_rows from FDCO for Dixon-Coles fitting
    # (odds_rows from CSV data have home_goals/away_goals; API data doesn't)
    train_all(X, y_result, y_goals, y_btts, odds_rows=odds_rows_fdco)


if __name__ == "__main__":
    asyncio.run(main())
