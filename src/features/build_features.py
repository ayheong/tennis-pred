# src/features/build_features.py
from __future__ import annotations
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Columns for modeling
NUM_DIFF_COLS = ["rank_diff", "age_diff", "ht_diff", "best_of", "hand_match"]
CAT_COLS = ["surface", "tourney_level"]

def load_matches(paths: list[str] | str) -> pd.DataFrame:
    """
    Read one or more CSVs and keep only useful columns.
    Accepts a glob pattern (str/Path) OR a list of file paths/patterns.
    """
    import os, glob
    from pathlib import Path

    # expand to a concrete file list
    if isinstance(paths, (str, Path)):
        patterns = [str(paths)]
    else:
        patterns = [str(p) for p in paths]

    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))

    if not files:
        raise FileNotFoundError(
            f"load_matches: no files matched {patterns} (cwd={os.getcwd()})"
        )

    frames = []
    use_cols = [
        "tourney_date", "surface", "tourney_level", "best_of",
        "winner_id","loser_id","winner_rank","winner_age","winner_ht","winner_hand",
        "loser_rank","loser_age","loser_ht","loser_hand",
        "winner_name","loser_name","round",
    ]

    for p in files:
        dfp = pd.read_csv(p, low_memory=False, usecols=lambda c: c in set(use_cols))
        frames.append(dfp)

    df = pd.concat(frames, ignore_index=True)

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")

    # minimal cleaning expected by feature builder
    df = df.dropna(subset=[
        "winner_id","loser_id","winner_rank","loser_rank",
        "surface","tourney_level","best_of","tourney_date"
    ])
    df["winner_id"] = df["winner_id"].astype(int)
    df["loser_id"]  = df["loser_id"].astype(int)
    df["best_of"]   = df["best_of"].astype(int)

    return df

def _row_to_feature_dict(r: pd.Series, flip: bool) -> tuple[dict, int]:
    """
    Build one feature dict from a single match row.
    If flip=True, we swap the perspective so label becomes 0 instead of 1.
    """
    # fill NaN with 0 for diffs
    w_age = 0.0 if pd.isna(r.get("winner_age")) else float(r["winner_age"])
    l_age = 0.0 if pd.isna(r.get("loser_age"))  else float(r["loser_age"])
    w_ht  = 0.0 if pd.isna(r.get("winner_ht"))  else float(r["winner_ht"])
    l_ht  = 0.0 if pd.isna(r.get("loser_ht"))   else float(r["loser_ht"])

    rank_diff = float(r["winner_rank"] - r["loser_rank"])
    age_diff  = w_age - l_age
    ht_diff   = w_ht  - l_ht
    hand_match = int(str(r.get("winner_hand","R")) == str(r.get("loser_hand","R")))

    feat = {
        "rank_diff": rank_diff,  # winner - loser rank
        "age_diff":  age_diff,  # winner - loser age
        "ht_diff":   ht_diff,  # winner - loser height
        "hand_match": hand_match,  # 1 if hands match, 0 otherwise
        "best_of":   int(r["best_of"]),  # number of sets per match
        "surface":   str(r["surface"]),  # hard, grass, clay, carpet
        "tourney_level": str(r["tourney_level"]),  # S for majors, M for 1000s, A for other tour level, C for challengers, etc
        "date":      r.get("tourney_date"),  # date of tourney
        "pid_a": int(r["winner_id"]),  # unique id for player a
        "pid_b": int(r["loser_id"]),  # unique id for player b
    }
    y = 1  # winner perspective

    if flip:
        # invert diffs
        feat["rank_diff"] *= -1
        feat["age_diff"]  *= -1
        feat["ht_diff"]   *= -1
        feat["pid_a"], feat["pid_b"] = feat["pid_b"], feat["pid_a"]
        y = 0

    return feat, y

def _compute_recency_weight(dates: pd.Series, half_life_days: int = 365) -> pd.Series:
    """Exponential decay weight relative to the most recent match date."""
    most_recent = pd.to_datetime(dates).max()
    age_days = (most_recent - pd.to_datetime(dates)).dt.days
    return 0.5 ** (age_days / half_life_days)

def add_last10_winrate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall last-10 match win rate per player.
    - Builds a long table of (player, match, is_win)
    - Rolling mean over last 10 matches, shifted by 1 (excludes current match)
    - Merges back to the wide match rows as w_wr10 / l_wr10
    - Fills missing with 0.5
    """
    df = df.copy().reset_index(drop=False).rename(columns={"index": "match_idx"})

    w = df[["match_idx","tourney_date","winner_id"]].assign(
        player_id=lambda x: x["winner_id"], is_win=1
    )[["match_idx","player_id","tourney_date","is_win"]]

    l = df[["match_idx","tourney_date","loser_id"]].assign(
        player_id=lambda x: x["loser_id"], is_win=0
    )[["match_idx","player_id","tourney_date","is_win"]]

    plays = pd.concat([w, l], ignore_index=True)
    plays = plays.sort_values(["player_id","tourney_date","match_idx"])

    # rolling last-10 win rate, shifted to avoid leakage
    plays["wr10"] = (plays.groupby("player_id", observed=True)["is_win"]
                          .transform(lambda s: s.rolling(10, min_periods=1).mean().shift(1)))

    # split back and merge to wide
    w_feats = plays.merge(df[["match_idx","winner_id"]], on="match_idx")
    w_feats = w_feats[w_feats["player_id"] == w_feats["winner_id"]][
        ["match_idx","wr10"]
    ].rename(columns={"wr10":"w_wr10"})

    l_feats = plays.merge(df[["match_idx","loser_id"]], on="match_idx")
    l_feats = l_feats[l_feats["player_id"] == l_feats["loser_id"]][
        ["match_idx","wr10"]
    ].rename(columns={"wr10":"l_wr10"})

    out = df.merge(w_feats, on="match_idx", how="left").merge(l_feats, on="match_idx", how="left")

    # early-career: no history yet → 0.5 prior
    out["w_wr10"] = out["w_wr10"].fillna(0.5)
    out["l_wr10"] = out["l_wr10"].fillna(0.5)
    return out

def add_h2h(df):
    df = df.sort_values("tourney_date").copy()
    h2h_key = list(zip(np.minimum(df["winner_id"], df["loser_id"]),
                       np.maximum(df["winner_id"], df["loser_id"])))
    df["_pair"] = h2h_key

    # label from winner perspective for this row
    df["_w_win"] = 1

    # build prior wins for (pair, player)
    prior_wins = {}
    w_rate = []
    for (_, row) in df.iterrows():
        a, b = row["_pair"]
        w = row["winner_id"]
        # from winner's perspective, prior wins he has vs the other:
        key = (a, b, w)
        total_ab = prior_wins.get((a, b, "total"), 0)
        wins_w = prior_wins.get(key, 0)
        # prior h2h rate (uninformative prior 0.5 if none)
        rate = wins_w / total_ab if total_ab > 0 else 0.5
        w_rate.append(rate)
        # update counts AFTER using them
        prior_wins[(a, b, "total")] = total_ab + 1
        prior_wins[key] = wins_w + 1

    df["w_h2h_prior"] = w_rate

    # loser’s prior h2h vs winner = 1 - winner’s (when any history)
    df["l_h2h_prior"] = 1 - df["w_h2h_prior"]
    return df.drop(columns=["_pair", "_w_win"])

def dataframe_to_examples(
    df: pd.DataFrame,
    flip_prob: float = 0.5,
    random_state: int = 42,
    half_life_days: int = 365
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert a matches dataframe into a features dataframe X and labels y,
    applying random perspective flips to avoid leakage and enforce symmetry.
    """
    df = add_last10_winrate(df)
    df = add_h2h(df)

    rng = np.random.default_rng(random_state)
    rows, labels = [], []
    for _, r in df.iterrows():
        flip = rng.random() < flip_prob
        feat, y = _row_to_feature_dict(r, flip=flip)

        wr10_diff = float(r["w_wr10"]) - float(r["l_wr10"])
        feat["wr10_diff"] = -wr10_diff if flip else wr10_diff

        h2h_prior_diff = float(r["w_h2h_prior"]) - float(r["l_h2h_prior"])
        feat["h2h_prior_diff"] = -h2h_prior_diff if flip else h2h_prior_diff

        rows.append(feat); labels.append(y)
    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="y", dtype=int)

    sample_weight = _compute_recency_weight(X["date"], half_life_days=half_life_days)
    # minimal NA handling on diffs
    for c in ["age_diff", "ht_diff", "wr10_diff", "h2h_prior_diff"]:
        X[c] = X[c].fillna(0.0)

    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12.0)

    X = X.drop(columns=["date", "month"])

    return X, y, sample_weight

def fit_label_encoders(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Fit LabelEncoders on categorical columns and return a mapping.
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    return X, encoders

def apply_label_encoders(X: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Apply existing encoders to categorical columns (for val/test/inference).
    Unknowns become -1 (LightGBM/XGBoost can handle if treated as missing).
    """
    X = X.copy()
    for col, le in encoders.items():
        # map unseen labels to -1
        known = {cls: idx for idx, cls in enumerate(le.classes_)}
        X[col] = X[col].astype(str).map(known).fillna(-1).astype(int)
    return X

def save_encoders(encoders: dict[str, LabelEncoder], path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {col: le.classes_.tolist() for col, le in encoders.items()}
    path.write_text(json.dumps(payload, indent=2))

def load_encoders(path: str | Path) -> dict[str, LabelEncoder]:
    payload = json.loads(Path(path).read_text())
    encoders: dict[str, LabelEncoder] = {}
    for col, classes in payload.items():
        le = LabelEncoder()
        le.classes_ = np.array(classes, dtype=object)
        encoders[col] = le
    return encoders
