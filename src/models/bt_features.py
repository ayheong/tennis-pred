# src/models/bt_features.py

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

def _exp_decay_weights(age_days: np.ndarray, half_life_days:365):
    return np.exp(-np.log(2.0) * (age_days / float(half_life_days)))

def fit_bt_strengths(matches: pd.DataFrame,
                     *,
                     as_of: pd.Timestamp,
                     surface: str | None = None,
                     half_life_days: float = 365.0,
                     C: float = 1.0,
                     max_iter: int = 300,
                     ) -> dict[int, float]:
    """Fit Bradley-Terry strengths on matches before given tourney date (as_of), can restrict by surface.
    Return dictionary mapping pid to strength scores (alpha values).
    """
    m = matches[matches["tourney_date"] < as_of].copy()
    if surface:
        m = m[matches["surface"] == surface]
    m = m.sort_values("tourney_date").reset_index(drop=True)

    age_in_days = (as_of - m["tourney_date"]).dt.days.to_numpy()
    w = _exp_decay_weights(age_in_days, half_life_days)  # recency weighted

    # create bt matrix
    pids = pd.Index(pd.unique(pd.concat([m["winner_id"], m["loser_id"]])))
    pid_to_col = {int(pid): i for i, pid in enumerate(pids)}
    n, p = len(m), len(pids)

    rows = np.arange(n)
    cols_win = m["winner_id"].map(pid_to_col).to_numpy()
    cols_loss = m["loser_id"].map(pid_to_col).to_numpy()
    X = sparse.csr_matrix((cols_win, cols_loss))
    y = np.ones(n, dtype=int)

    # fit logistic regression
    lr = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=max_iter, n_jobs=1).fit(X, y, sample_weight=w)
    alphas = lr.coef_.ravel()

    return {int(pid): float(alpha) for pid, alpha in zip(pids, alphas)}

def bt_diff(bt_scores_map: dict[int, float], pid_a: int, pid_b: int, default: float = 0.0) -> float:
    return float(bt_scores_map.get(int(pid_a), default) - bt_scores_map.get(int(pid_b), default))


