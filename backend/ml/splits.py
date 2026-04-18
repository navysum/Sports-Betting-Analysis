"""
Chronological data split constants — shared by train.py and optimize_blend.py.

Why this exists
---------------
Previously train.py used a 80/20 split to fit calibrators on the last 20% of
data, and optimize_blend.py evaluated blend weights on the same last 20%.
Because the calibrators had already been fit to every sample in that window,
the Brier scores being minimised during blend search were in-sample for the
calibrators. The "optimal" blend weights were overfit to the calibration set.

The fix is a three-way chronological split:
  •  [0 .. TRAIN_END)       → train XGBoost
  •  [TRAIN_END .. CAL_END) → fit probability calibrators
  •  [CAL_END .. 1.0)       → optimise DC/XGB blend weights

Both files import these constants, so there is a single source of truth.
Assumes the input X/y arrays are already chronologically ordered (they are —
build_fdco_training_data() iterates SEASONS + sorts each CSV by date).
"""

# Fractions define split boundaries, cumulative from 0.
TRAIN_FRACTION     = 0.70   # first 70% → XGBoost training
CALIBRATION_FRACTION = 0.15 # next 15% → probability calibration
BLEND_FRACTION     = 0.15   # last 15% → blend weight search

# Sanity check: fractions must sum to 1.0.
assert abs((TRAIN_FRACTION + CALIBRATION_FRACTION + BLEND_FRACTION) - 1.0) < 1e-9, (
    "Split fractions must sum to 1.0"
)

# Cumulative boundaries — easier to use directly.
TRAIN_END = TRAIN_FRACTION                               # 0.70
CAL_END   = TRAIN_FRACTION + CALIBRATION_FRACTION        # 0.85
# BLEND_END implicit at 1.0


def split_indices(n: int) -> tuple[int, int]:
    """
    Return (train_end_idx, cal_end_idx) for a dataset of size n.

    Usage:
        train_end, cal_end = split_indices(len(X))
        X_train, X_cal, X_blend = X[:train_end], X[train_end:cal_end], X[cal_end:]
    """
    train_end = int(n * TRAIN_END)
    cal_end   = int(n * CAL_END)
    # Guarantee at least 1 sample in each partition — degenerate safety net.
    train_end = max(1, min(train_end, n - 2))
    cal_end   = max(train_end + 1, min(cal_end, n - 1))
    return train_end, cal_end
