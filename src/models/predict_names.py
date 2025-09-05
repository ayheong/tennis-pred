# src/models/predict_names.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.build_features import apply_label_encoders, load_matches
from src.models.predict import load_artifacts

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"

# ---- Utilities --------------------------------------------------------
def _normalize_surface(s: object) -> str:
    """Normalize surface labels to Title case (e.g., 'Hard', 'Clay', 'Grass')."""
    return str(s).strip().title()

def _latest_known(s: pd.Series) -> float | str | np.floating | np.integer | pd.Timestamp | np.nan:
    """Return last non-NA value (or NaN if none)."""
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan

def _plays_for(df: pd.DataFrame, pid: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return matches before as_of for player id with basic flags."""
    d = df[df["tourney_date"] < as_of].copy()
    d = d[(d["winner_id"] == pid) | (d["loser_id"] == pid)]
    if d.empty:
        return d
    d["is_win"] = (d["winner_id"] == pid).astype(int)
    if "surface_n" not in d.columns:
        d["surface_n"] = d["surface"].map(_normalize_surface)
    # stable secondary sort to preserve original order within date
    d = d.reset_index().sort_values(["tourney_date", "index"], kind="mergesort")
    return d[["tourney_date", "surface_n", "is_win"]]

def _wr_last_k(df: pd.DataFrame, pid: int, as_of: pd.Timestamp, k: int = 10, surface: Optional[str] = None) -> float:
    """K-match recent win rate with simple shrinkage toward 0.5 when <k samples."""
    p = _plays_for(df, pid, as_of)
    if p.empty:
        return 0.5
    if surface is not None:
        p = p[p["surface_n"] == _normalize_surface(surface)]
    tail = p["is_win"].tail(k)
    n = len(tail)
    if n == 0:
        return 0.5
    # shrink to 0.5 when fewer than k available
    return float((tail.mean() * n + 0.5 * (k - n)) / k)

def _wr_window_days(df: pd.DataFrame, pid: int, as_of: pd.Timestamp, days: int = 365, surface: Optional[str] = None) -> float:
    """Windowed win rate over the last `days`."""
    p = _plays_for(df, pid, as_of)
    if p.empty:
        return 0.5
    start = as_of - pd.Timedelta(days=days)
    p = p[p["tourney_date"] >= start]
    if surface is not None:
        p = p[p["surface_n"] == _normalize_surface(surface)]
    if p.empty:
        return 0.5
    return float(p["is_win"].mean())

def _h2h_prior(df: pd.DataFrame, a: int, b: int, as_of: pd.Timestamp) -> Tuple[float, float]:
    """Smoothed H2H rate (Beta(1,1) prior). Returns (a_rate, b_rate)."""
    d = df[df["tourney_date"] < as_of]
    ab = d[((d["winner_id"] == a) & (d["loser_id"] == b)) | ((d["winner_id"] == b) & (d["loser_id"] == a))]
    if ab.empty:
        return 0.5, 0.5
    a_wins, total = (ab["winner_id"] == a).sum(), len(ab)
    a_rate = (a_wins + 1) / (total + 2)
    return float(a_rate), float(1 - a_rate)

def _h2h_counts(df: pd.DataFrame, a: int, b: int, as_of: pd.Timestamp, surface: Optional[str] = None) -> Dict[str, int]:
    """Raw H2H counts overall or on a surface."""
    d = df[df["tourney_date"] < as_of]
    if surface is not None:
        d = d[d["surface_n"] == _normalize_surface(surface)]
    ab = d[((d["winner_id"] == a) & (d["loser_id"] == b)) | ((d["winner_id"] == b) & (d["loser_id"] == a))]
    a_w = int((ab["winner_id"] == a).sum()); b_w = int((ab["winner_id"] == b).sum())
    return {"a_wins": a_w, "b_wins": b_w, "total": int(len(ab))}

def _build_name_maps(big: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """Build name→id map from a consolidated matches frame."""
    df = big[["winner_name", "winner_id", "loser_name", "loser_id"]].copy()
    for col in ("winner_name", "loser_name"):
        df[col] = df[col].astype(str).str.strip()
    pairs = pd.concat(
        [
            df[["winner_name", "winner_id"]].rename(columns={"winner_name": "name", "winner_id": "id"}),
            df[["loser_name", "loser_id"]].rename(columns={"loser_name": "name", "loser_id": "id"}),
        ],
        ignore_index=True,
    ).dropna()
    idx = (
        pairs.groupby(["name", "id"]).size().reset_index(name="n")
        .sort_values(["name", "n"], ascending=[True, False])
    )
    best = idx.drop_duplicates("name")
    name2id = {n.lower(): int(i) for n, i in zip(best["name"], best["id"])}
    return name2id, list(best["name"])

def _resolve_name(name: str, name2id: Dict[str, int], all_names: List[str]) -> Tuple[int, str]:
    """Exact match first, then fuzzy via difflib with a reasonable cutoff."""
    key = name.strip().lower()
    if key in name2id:
        return name2id[key], name
    hit = get_close_matches(name, all_names, n=1, cutoff=0.75)
    if hit:
        return name2id[hit[0].lower()], hit[0]
    raise ValueError(f"Player not found: {name}")

def _player_snapshot(df: pd.DataFrame, pid: int, as_of: pd.Timestamp) -> Dict[str, object]:
    """Recent static info for a player prior to as_of."""
    d = df[df["tourney_date"] < as_of]
    w, l = d[d["winner_id"] == pid], d[d["loser_id"] == pid]
    rank = _latest_known(pd.concat([w["winner_rank"], l["loser_rank"]], ignore_index=True))
    age  = _latest_known(pd.concat([w["winner_age"],  l["loser_age"]],  ignore_index=True))
    ht   = _latest_known(pd.concat([w["winner_ht"],   l["loser_ht"]],   ignore_index=True))
    hand = _latest_known(pd.concat([w["winner_hand"], l["loser_hand"]], ignore_index=True))
    return {"rank": rank, "age": age, "ht": ht, "hand": hand}

def _build_match_row(
    df: pd.DataFrame,
    pid_a: int,
    pid_b: int,
    as_of: pd.Timestamp,
    surface: str,
    best_of: int,
    level: str,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object], float, float]:
    """Assemble one-row feature frame and auxiliary info."""
    A, B = _player_snapshot(df, pid_a, as_of), _player_snapshot(df, pid_b, as_of)
    a_h2h, b_h2h = _h2h_prior(df, pid_a, pid_b, as_of)
    A_wr10 = _wr_last_k(df, pid_a, as_of, k=10)
    B_wr10 = _wr_last_k(df, pid_b, as_of, k=10)

    # Safely compute diffs (treat missing as 0)
    rank_a = A["rank"] if pd.notna(A["rank"]) else 0
    rank_b = B["rank"] if pd.notna(B["rank"]) else 0
    rank_diff = float(rank_a - rank_b)

    age_diff  = float((A["age"] or 0) - (B["age"] or 0))
    ht_diff   = float((A["ht"]  or 0) - (B["ht"]  or 0))
    hand_match = int(str(A["hand"] or "R") == str(B["hand"] or "R"))
    wr10_diff = float(A_wr10 - B_wr10)
    h2h_prior_diff = float(a_h2h - b_h2h)

    year = as_of.year
    m = as_of.month
    X = pd.DataFrame(
        [
            {
                "rank_diff": rank_diff,
                "age_diff": age_diff,
                "ht_diff": ht_diff,
                "hand_match": hand_match,
                "best_of": int(best_of),
                "surface": str(surface),
                "tourney_level": str(level),
                "wr10_diff": wr10_diff,
                "h2h_prior_diff": h2h_prior_diff,
                "year": year,
                "month_sin": float(np.sin(2 * np.pi * m / 12.0)),
                "month_cos": float(np.cos(2 * np.pi * m / 12.0)),
            }
        ]
    )
    return X, A, B, a_h2h, b_h2h

# ---- Public API -------------------------------------------------------------

def predict_by_names(
    name_a: str,
    name_b: str,
    surface: str = "Hard",
    best_of: int = 3,
    tourney_level: str = "A",
    as_of: Optional[str] = None,
) -> dict:
    """
    Predict match outcome A vs B with lightweight features & model ensemble.

    Returns a JSON-serializable dict with probabilities, pick, and insights.
    """
    # Load data once, in two forms:
    # 1) `df` used by feature builder (your engineered schema via load_matches)
    # 2) `big` used for name maps / raw counts / recent form; keep only needed cols to save RAM
    files = sorted((DATA_DIR).glob("atp_matches_202*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs in {DATA_DIR}/atp_matches_202*.csv")
    paths = [str(p) for p in files]

    # Engineered (same as training)
    df = load_matches(paths)
    df = df[df["tourney_date"] >= pd.Timestamp("2000-01-01")].copy()

    # Lightweight raw view for names/H2H; parse only needed columns
    use_cols = [
        "tourney_date",
        "surface",
        "winner_id", "loser_id",
        "winner_name", "loser_name",
        "winner_rank", "loser_rank",
        "winner_age", "loser_age",
        "winner_ht", "loser_ht",
        "winner_hand", "loser_hand",
    ]
    parts = [
        pd.read_csv(p, low_memory=False, usecols=lambda c, _set=set(use_cols): c in _set)
        for p in paths
    ]
    big = pd.concat(parts, ignore_index=True)
    big["tourney_date"] = pd.to_datetime(big["tourney_date"], format="%Y%m%d", errors="coerce")
    big["surface_n"] = big["surface"].map(_normalize_surface)

    # Name resolution (exact → fuzzy)
    name2id, all_names = _build_name_maps(big)
    pid_a, disp_a = _resolve_name(name_a, name2id, all_names)
    pid_b, disp_b = _resolve_name(name_b, name2id, all_names)

    # as_of default: one day after the latest engineered row to ensure "past-only"
    as_of_ts = pd.to_datetime(as_of) if as_of else (df["tourney_date"].max() + pd.Timedelta(days=1))

    # Build features
    X_row, A_snap, B_snap, a_h2h_rate, b_h2h_rate = _build_match_row(
        big, pid_a, pid_b, as_of_ts, surface, best_of, tourney_level
    )

    # Encode & predict
    models, encs, thr = load_artifacts()
    X_enc = apply_label_encoders(X_row.copy(), encs)
    for c in ("surface", "tourney_level"):
        if c in X_enc.columns:
            X_enc[c] = X_enc[c].astype("category")

    # Average ensemble proba (class 1 = A wins)
    preds = [m.predict_proba(X_enc)[:, 1] for m in models]
    p = float(np.mean(preds, axis=0))
    pick_is_A = p >= float(thr)

    # Insights (use big which has all rows)
    surf_key = _normalize_surface(surface)
    h2h_all  = _h2h_counts(big, pid_a, pid_b, as_of_ts, surface=None)
    h2h_surf = _h2h_counts(big, pid_a, pid_b, as_of_ts, surface=surf_key)

    A_wr10       = _wr_last_k(big, pid_a, as_of_ts, k=10)
    B_wr10       = _wr_last_k(big, pid_b, as_of_ts, k=10)
    A_wr10_surf  = _wr_last_k(big, pid_a, as_of_ts, k=10, surface=surf_key)
    B_wr10_surf  = _wr_last_k(big, pid_b, as_of_ts, k=10, surface=surf_key)
    A_wr365_all  = _wr_window_days(big, pid_a, as_of_ts, days=365)
    B_wr365_all  = _wr_window_days(big, pid_b, as_of_ts, days=365)
    A_wr365_surf = _wr_window_days(big, pid_a, as_of_ts, days=365, surface=surf_key)
    B_wr365_surf = _wr_window_days(big, pid_b, as_of_ts, days=365, surface=surf_key)

    return {
        "players": {"A": disp_a, "B": disp_b},
        "as_of": str(as_of_ts.date()),
        "inputs": {"surface": surface, "best_of": best_of, "tourney_level": tourney_level},
        "prob_A_wins": p,
        "threshold": float(thr),
        "pick": disp_a if pick_is_A else disp_b,
        "pick_is_A": bool(pick_is_A),
        "insights": {
            "h2h_overall": {
                "A_wins": h2h_all["a_wins"],
                "B_wins": h2h_all["b_wins"],
                "total": h2h_all["total"],
                "A_rate_smoothed": round(a_h2h_rate, 3),
            },
            "h2h_on_surface": {
                "A_wins": h2h_surf["a_wins"],
                "B_wins": h2h_surf["b_wins"],
                "total": h2h_surf["total"],
            },
            "recent_form": {
                "A_wr10": round(A_wr10, 3),
                "B_wr10": round(B_wr10, 3),
                f"A_wr10_{surf_key}": round(A_wr10_surf, 3),
                f"B_wr10_{surf_key}": round(B_wr10_surf, 3),
                "A_wr365_all": round(A_wr365_all, 3),
                "B_wr365_all": round(B_wr365_all, 3),
                f"A_wr365_{surf_key}": round(A_wr365_surf, 3),
                f"B_wr365_{surf_key}": round(B_wr365_surf, 3),
            },
            "player_info": {
                "A": {"rank": A_snap["rank"], "age": A_snap["age"], "ht": A_snap["ht"], "hand": A_snap["hand"]},
                "B": {"rank": B_snap["rank"], "age": B_snap["age"], "ht": B_snap["ht"], "hand": B_snap["hand"]},
            },
        },
    }

# ---- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.models.predict_names 'Player A' 'Player B' [Surface] [BestOf] [Level] [YYYY-MM-DD]")
        raise SystemExit(1)
    name_a, name_b = sys.argv[1], sys.argv[2]
    surface = sys.argv[3] if len(sys.argv) > 3 else "Hard"
    best_of = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    level   = sys.argv[5] if len(sys.argv) > 5 else "A"
    as_of   = sys.argv[6] if len(sys.argv) > 6 else None
    print(json.dumps(predict_by_names(name_a, name_b, surface, best_of, level, as_of), indent=2))
