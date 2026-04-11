"""
Train three models: 1X2 result, Over/Under 2.5 goals, BTTS.

Run from backend/ directory:
    python -m ml.train

Models saved to:
    ml/result_model.joblib
    ml/goals_model.joblib
    ml/btts_model.joblib
    ml/training_log.json
"""
import asyncio
import os
import json
import time
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from ml.features import build_feature_vector, FEATURE_NAMES, N_FEATURES

ML_DIR = os.path.dirname(__file__)
RESULT_MODEL_PATH  = os.path.join(ML_DIR, "result_model.joblib")
GOALS_MODEL_PATH   = os.path.join(ML_DIR, "goals_model.joblib")
BTTS_MODEL_PATH    = os.path.join(ML_DIR, "btts_model.joblib")
TRAINING_LOG_PATH  = os.path.join(ML_DIR, "training_log.json")

COMPETITIONS = ["PL", "PD", "BL1", "SA", "FL1", "ELC", "DED"]


def _make_xgb(n_classes: int = 3) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=3,
        gamma=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        num_class=n_classes if n_classes > 2 else None,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        random_state=42,
        n_jobs=-1,
    )


async def fetch_training_data():
    """
    Pull finished matches and build:
      X           — feature matrix  (N, 28)
      y_result    — 0=HOME, 1=DRAW, 2=AWAY
      y_goals     — 0=Under2.5, 1=Over2.5
      y_btts      — 0=No, 1=Yes
    """
    from app.services.football_api import get_finished_matches, get_team_matches, get_standings

    X, y_result, y_goals, y_btts = [], [], [], []
    skipped = 0

    for comp in COMPETITIONS:
        print(f"  [{comp}] Fetching matches...")
        try:
            matches = await get_finished_matches(comp, limit=380)
            standings_table = await get_standings(comp)
            standing_map = {row["team"]["id"]: row for row in standings_table}
        except Exception as e:
            print(f"  [{comp}] Error: {e}")
            continue

        finished = [m for m in matches if m.get("status") == "FINISHED"]
        print(f"  [{comp}] {len(finished)} finished matches")

        for i, match in enumerate(finished):
            home_id = match.get("homeTeam", {}).get("id")
            away_id = match.get("awayTeam", {}).get("id")
            hg = match.get("score", {}).get("fullTime", {}).get("home")
            ag = match.get("score", {}).get("fullTime", {}).get("away")

            if not all([home_id, away_id, hg is not None, ag is not None]):
                skipped += 1
                continue

            # Labels
            if hg > ag:   result_label = 0  # HOME
            elif hg == ag: result_label = 1  # DRAW
            else:          result_label = 2  # AWAY

            goals_label = 1 if (hg + ag) > 2 else 0
            btts_label  = 1 if (hg > 0 and ag > 0) else 0

            try:
                home_m, away_m = await asyncio.gather(
                    get_team_matches(home_id, limit=25),
                    get_team_matches(away_id, limit=25),
                )
            except Exception:
                skipped += 1
                continue

            vec = build_feature_vector(
                home_id, away_id,
                home_m, away_m,
                h2h_matches=[],
                home_standing=standing_map.get(home_id),
                away_standing=standing_map.get(away_id),
            )
            X.append(vec)
            y_result.append(result_label)
            y_goals.append(goals_label)
            y_btts.append(btts_label)

            # Rate-limit respect — pause every 10 matches
            if (i + 1) % 10 == 0:
                await asyncio.sleep(6)

    print(f"\nTotal samples: {len(X)} | Skipped: {skipped}")
    return (
        np.array(X, dtype=np.float32),
        np.array(y_result),
        np.array(y_goals),
        np.array(y_btts),
    )


def _train_and_eval(model: XGBClassifier, X: np.ndarray, y: np.ndarray, label: str) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"  {label} CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    model.fit(X, y)
    return {"accuracy_mean": float(scores.mean()), "accuracy_std": float(scores.std())}


def train_all(
    X: np.ndarray,
    y_result: np.ndarray,
    y_goals: np.ndarray,
    y_btts: np.ndarray,
) -> dict:
    print("\nTraining result model (1X2)...")
    result_model = _make_xgb(n_classes=3)
    result_stats = _train_and_eval(result_model, X, y_result, "Result")
    joblib.dump(result_model, RESULT_MODEL_PATH)

    print("Training goals model (O/U 2.5)...")
    goals_model = _make_xgb(n_classes=2)
    goals_stats = _train_and_eval(goals_model, X, y_goals, "Goals")
    joblib.dump(goals_model, GOALS_MODEL_PATH)

    print("Training BTTS model...")
    btts_model = _make_xgb(n_classes=2)
    btts_stats = _train_and_eval(btts_model, X, y_btts, "BTTS")
    joblib.dump(btts_model, BTTS_MODEL_PATH)

    log = _load_training_log()
    entry = {
        "trained_at": __import__("datetime").datetime.utcnow().isoformat(),
        "samples": int(len(X)),
        "result_model": result_stats,
        "goals_model": goals_stats,
        "btts_model": btts_stats,
        "feature_count": N_FEATURES,
        "features": FEATURE_NAMES,
    }
    log.append(entry)
    _save_training_log(log)

    print(f"\nAll models saved. Training log updated ({len(log)} entries).")
    return entry


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


async def main():
    print("=== Sports Bet Analysis — Model Training ===")
    print("Fetching training data (this takes several minutes due to API rate limits)...\n")
    X, y_result, y_goals, y_btts = await fetch_training_data()

    if len(X) < 50:
        print("ERROR: Not enough training data. Check your API key and try again.")
        return

    train_all(X, y_result, y_goals, y_btts)


if __name__ == "__main__":
    asyncio.run(main())
