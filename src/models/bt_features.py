# src/models/bt_features.py

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

def _exp_decay_weights(age_days: np.ndarray, half_life_days: float = 365):
    return np.exp(-np.log(2.0) * (age_days / float(half_life_days)))

def fit_bt_strengths(matches: pd.DataFrame,
                     *,
                     as_of: str | pd.Timestamp,
                     surface: str | None = None,
                     half_life_days: float = 365.0,
                     C: float = 1.0,
                     max_iter: int = 300,
                     ) -> dict[int, float]:
    """
    Fit Bradley-Terry strengths on matches before given tourney date (as_of), can restrict by surface.
    Return dictionary mapping pid to strength scores (alpha values).
    """
    as_of = pd.to_datetime(as_of)

    m = matches[matches["tourney_date"] < as_of].copy()

    if m.empty:
        return {}

    if surface:
        m = m[m["surface"] == surface]
    m = m.sort_values("tourney_date").reset_index(drop=True)

    age_in_days = (as_of - m["tourney_date"]).dt.days.to_numpy()
    w = _exp_decay_weights(age_in_days, half_life_days)  # recency weighted

    # create bt matrix
    pids = pd.Index(pd.unique(pd.concat([m["winner_id"], m["loser_id"]])))
    pid_to_col = {int(pid): i for i, pid in enumerate(pids)}
    n, p = len(m), len(pids)

    rows = np.arange(n)
    cols_win = m["winner_id"].map(pid_to_col).to_numpy()  # [match_0: winner id, match_1: winner_id, ..., match_n: winner_id]
    cols_loss = m["loser_id"].map(pid_to_col).to_numpy()  # [match_0: loser_id, match_1: loser_id, ..., match_n: loser_id]

    row_idx = np.r_[rows, rows]  # [match_0, match_1, ..., match_n, match_0, match_1, ..., match_n]
    col_idx = np.r_[cols_win, cols_loss]  # [winner_ids, loser_ids]
    data = np.r_[np.ones(n), -np.ones(n)]  # [+1 for winners, -1 for losers]

    # example matrix
    # |          player_0, player_1, player_2, ..., player_p |
    # | match_0     +1        -1        0      ...      0    |  match_0: player 0 beats player 1
    # | match_1      0         0        0      ...      0    |
    # | match_2     ...       ...      ...     ...     ...   |
    # | ...         ...       ...      ...     ...     ...   |
    # | match_n     ...       ...      ...     ...     ...   |
    X = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, p))
    y = np.ones(n, dtype=int)  # y = 1: event occured, i.e. player_a beat player_b

    # need two classes (not just y=1) for LogisticRegression to function, create a mirrored version of matrix X and append
    X2 = sparse.vstack((X, -X), format="csr")
    y2 = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]
    w2 = np.r_[w, w]

    # fit logistic regression
    lr = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=max_iter, n_jobs=1, fit_intercept=False).fit(X2, y2, sample_weight=w2)
    alphas = lr.coef_.ravel()  # list of strength scores, larger alpha = stronger player comparatively
    alphas -= alphas.mean()  # center on mean for readability

    return {int(pid): float(alpha) for pid, alpha in zip(pids, alphas)}

def bt_diff(bt_scores_map: dict[int, float], pid_a: int, pid_b: int, default: float = 0.0) -> float:
    """
    return difference of strength scores between player_a and player_b
    """
    return float(bt_scores_map.get(int(pid_a), default) - bt_scores_map.get(int(pid_b), default))


