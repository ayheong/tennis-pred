# src/models/predict_names.py
import json, joblib, numpy as np, pandas as pd, glob
from pathlib import Path
from difflib import get_close_matches

from src.features.build_features import apply_label_encoders, load_encoders, load_matches
from src.models.predict import load_artifacts

# resolve project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"

def build_name_index(pattern="atp_matches_202*.csv"):
    # build name→id table from files
    use_cols = ["winner_id","loser_id","winner_name","loser_name","tourney_date"]
    files = sorted((DATA_DIR).glob(pattern))
    if not files: raise FileNotFoundError(f"no files matched in {DATA_DIR}/{pattern}")
    frames = [pd.read_csv(p, low_memory=False, usecols=lambda c: c in set(use_cols)) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df["winner_name"] = df["winner_name"].astype(str).str.strip()
    df["loser_name"]  = df["loser_name"].astype(str).str.strip()
    pairs = pd.concat([
        df[["winner_name","winner_id"]].rename(columns={"winner_name":"name","winner_id":"id"}),
        df[["loser_name","loser_id"]].rename(columns={"loser_name":"name","loser_id":"id"})
    ], ignore_index=True).dropna()
    idx = (pairs.groupby(["name","id"]).size().reset_index(name="n")
                 .sort_values(["name","n"], ascending=[True,False]))
    best = idx.drop_duplicates("name")
    name2id = dict(zip(best["name"].str.lower(), best["id"].astype(int)))
    all_names = list(best["name"])
    return name2id, all_names

def _normalize_surface(s):
    # normalize surface labels
    return str(s).strip().title()

def _latest_known(s):
    # last non-na
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan

def _player_snapshot(df, pid, as_of):
    # pull recent static info
    d = df[df["tourney_date"] < as_of]
    w, l = d[d["winner_id"] == pid], d[d["loser_id"] == pid]
    rank = _latest_known(pd.concat([w["winner_rank"], l["loser_rank"]], ignore_index=True))
    age  = _latest_known(pd.concat([w["winner_age"],  l["loser_age"]],  ignore_index=True))
    ht   = _latest_known(pd.concat([w["winner_ht"],   l["loser_ht"]],   ignore_index=True))
    hand = _latest_known(pd.concat([w["winner_hand"], l["loser_hand"]], ignore_index=True))
    return dict(rank=rank, age=age, ht=ht, hand=hand)

def _plays_for(df, pid, as_of):
    # matches before as_of for pid
    d = df[df["tourney_date"] < as_of].copy()
    d = d[(d["winner_id"] == pid) | (d["loser_id"] == pid)].copy()
    if d.empty: return d
    d["is_win"] = (d["winner_id"] == pid).astype(int)
    d["surface_n"] = d["surface"].map(_normalize_surface)
    d = d.reset_index().sort_values(["tourney_date","index"], kind="mergesort")
    return d[["tourney_date","surface_n","is_win"]]

def _wr_last_k(df, pid, as_of, k=10, surface=None):
    # last-k win rate (shrunk)
    p = _plays_for(df, pid, as_of)
    if p.empty: return 0.5
    if surface is not None:
        p = p[p["surface_n"] == _normalize_surface(surface)]
    tail = p["is_win"].tail(k)
    n = len(tail)
    if n == 0: return 0.5
    return float((tail.mean() * n + 0.5 * (k - n)) / k)

def _wr_window_days(df, pid, as_of, days=365, surface=None):
    # window win rate
    p = _plays_for(df, pid, as_of)
    if p.empty: return 0.5
    start = as_of - pd.Timedelta(days=days)
    p = p[p["tourney_date"] >= start]
    if surface is not None:
        p = p[p["surface_n"] == _normalize_surface(surface)]
    if len(p) == 0: return 0.5
    return float(p["is_win"].mean())

def _h2h_prior(df, a, b, as_of):
    # smoothed h2h rate
    d = df[df["tourney_date"] < as_of]
    ab = d[((d["winner_id"] == a) & (d["loser_id"] == b)) | ((d["winner_id"] == b) & (d["loser_id"] == a))]
    if ab.empty: return 0.5, 0.5
    a_wins, total = (ab["winner_id"] == a).sum(), len(ab)
    a_rate = (a_wins + 1) / (total + 2)
    return float(a_rate), float(1 - a_rate)

def _h2h_counts(df, a, b, as_of, surface=None):
    # raw h2h counts (optional surface)
    d = df[df["tourney_date"] < as_of].copy()
    if surface is not None:
        d["surface_n"] = d["surface"].map(_normalize_surface)
        d = d[d["surface_n"] == _normalize_surface(surface)]
    ab = d[((d["winner_id"] == a) & (d["loser_id"] == b)) | ((d["winner_id"] == b) & (d["loser_id"] == a))]
    a_w = int((ab["winner_id"] == a).sum()); b_w = int((ab["winner_id"] == b).sum())
    return dict(a_wins=a_w, b_wins=b_w, total=int(len(ab)))

def _name_maps(df):
    # quick name→id map from a big df
    df = df.copy()
    df["winner_name"] = df.get("winner_name", "").astype(str).str.strip()
    df["loser_name"]  = df.get("loser_name", "").astype(str).str.strip()
    pairs = pd.concat([
        df[["winner_name","winner_id"]].rename(columns={"winner_name":"name","winner_id":"id"}),
        df[["loser_name","loser_id"]].rename(columns={"loser_name":"name","loser_id":"id"}),
    ], ignore_index=True).dropna()
    idx = (pairs.groupby(["name","id"]).size().reset_index(name="n")
                 .sort_values(["name","n"], ascending=[True,False]))
    best = idx.drop_duplicates("name")
    return {n.lower(): int(i) for n,i in zip(best["name"], best["id"])}, list(best["name"])

def _resolve(name, name2id, all_names):
    # exact then fuzzy
    key = name.strip().lower()
    if key in name2id: return name2id[key], name
    hit = get_close_matches(name, all_names, n=1, cutoff=0.75)
    if hit: return name2id[hit[0].lower()], hit[0]
    raise ValueError(f"Player not found: {name}")

def _build_match_row(df, pid_a, pid_b, as_of, surface, best_of, level):
    # one-row features for prediction
    A, B = _player_snapshot(df, pid_a, as_of), _player_snapshot(df, pid_b, as_of)
    a_h2h, b_h2h = _h2h_prior(df, pid_a, pid_b, as_of)
    A_wr10 = _wr_last_k(df, pid_a, as_of, k=10, surface=None)
    B_wr10 = _wr_last_k(df, pid_b, as_of, k=10, surface=None)
    rank_diff = float((A["rank"] if pd.notna(A["rank"]) else 0) - (B["rank"] if pd.notna(B["rank"]) else 0))
    age_diff  = float((A["age"] or 0) - (B["age"] or 0))
    ht_diff   = float((A["ht"]  or 0) - (B["ht"]  or 0))
    hand_match = int(str(A["hand"] or "R") == str(B["hand"] or "R"))
    wr10_diff = float(A_wr10 - B_wr10)
    h2h_prior_diff = float(a_h2h - b_h2h)
    year = as_of.year; m = as_of.month
    X = pd.DataFrame([{
        "rank_diff": rank_diff, "age_diff": age_diff, "ht_diff": ht_diff,
        "hand_match": hand_match, "best_of": int(best_of),
        "surface": str(surface), "tourney_level": str(level),
        "wr10_diff": wr10_diff, "h2h_prior_diff": h2h_prior_diff,
        "year": year,
        "month_sin": float(np.sin(2*np.pi*m/12.0)),
        "month_cos": float(np.cos(2*np.pi*m/12.0)),
    }])
    return X, A, B, a_h2h, b_h2h

def predict_by_names(name_a, name_b, surface="Hard", best_of=3, tourney_level="A", as_of=None):
    # load the same files used in training
    files = sorted(DATA_DIR.glob("atp_matches_202*.csv"))
    if not files: raise FileNotFoundError(f"no CSVs in {DATA_DIR}/atp_matches_202*.csv")
    paths = [str(p) for p in files]

    # parsed/filtered for features
    df = load_matches(paths)
    df = df[df["tourney_date"] >= pd.Timestamp("2000-01-01")]

    # big df for names/h2h/surface
    big = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    big["tourney_date"] = pd.to_datetime(big["tourney_date"], format="%Y%m%d", errors="coerce")

    # name resolution
    name2id, all_names = _name_maps(big)
    pid_a, disp_a = _resolve(name_a, name2id, all_names)
    pid_b, disp_b = _resolve(name_b, name2id, all_names)

    # as-of date
    as_of = pd.to_datetime(as_of) if as_of else (df["tourney_date"].max() + pd.Timedelta(days=1))

    # build one-row features
    X_row, A_snap, B_snap, a_h2h_rate, b_h2h_rate = _build_match_row(
        df, pid_a, pid_b, as_of, surface, best_of, tourney_level
    )

    # encode and predict
    models, encs, thr = load_artifacts()
    X_enc = apply_label_encoders(X_row.copy(), encs)
    for c in ["surface","tourney_level"]:
        if c in X_enc.columns: X_enc[c] = X_enc[c].astype("category")
    p = float(np.mean([m.predict_proba(X_enc)[:,1] for m in models], axis=0))
    pick_is_A = p >= thr

    # insights (use big for counts/form; it has all rows)
    surf_key = _normalize_surface(surface)
    h2h_all  = _h2h_counts(big, pid_a, pid_b, as_of, surface=None)
    h2h_surf = _h2h_counts(big, pid_a, pid_b, as_of, surface=surf_key)

    A_wr10       = _wr_last_k(big, pid_a, as_of, k=10)
    B_wr10       = _wr_last_k(big, pid_b, as_of, k=10)
    A_wr10_surf  = _wr_last_k(big, pid_a, as_of, k=10, surface=surf_key)
    B_wr10_surf  = _wr_last_k(big, pid_b, as_of, k=10, surface=surf_key)
    A_wr365_all  = _wr_window_days(big, pid_a, as_of, days=365)
    B_wr365_all  = _wr_window_days(big, pid_b, as_of, days=365)
    A_wr365_surf = _wr_window_days(big, pid_a, as_of, days=365, surface=surf_key)
    B_wr365_surf = _wr_window_days(big, pid_b, as_of, days=365, surface=surf_key)

    return {
        "players": {"A": disp_a, "B": disp_b},
        "as_of": str(as_of.date()),
        "inputs": {"surface": surface, "best_of": best_of, "tourney_level": tourney_level},
        "prob_A_wins": p,
        "threshold": float(thr),
        "pick": disp_a if pick_is_A else disp_b,
        "pick_is_A": bool(pick_is_A),
        "insights": {
            "h2h_overall": {
                "A_wins": h2h_all["a_wins"], "B_wins": h2h_all["b_wins"],
                "total": h2h_all["total"], "A_rate_smoothed": round(a_h2h_rate, 3)
            },
            "h2h_on_surface": {
                "A_wins": h2h_surf["a_wins"], "B_wins": h2h_surf["b_wins"],
                "total": h2h_surf["total"]
            },
            "recent_form": {
                "A_wr10": round(A_wr10, 3), "B_wr10": round(B_wr10, 3),
                f"A_wr10_{surf_key}": round(A_wr10_surf, 3),
                f"B_wr10_{surf_key}": round(B_wr10_surf, 3),
                "A_wr365_all": round(A_wr365_all, 3), "B_wr365_all": round(B_wr365_all, 3),
                f"A_wr365_{surf_key}": round(A_wr365_surf, 3),
                f"B_wr365_{surf_key}": round(B_wr365_surf, 3),
            },
            "player_info": {
                "A": {"rank": A_snap["rank"], "age": A_snap["age"], "ht": A_snap["ht"], "hand": A_snap["hand"]},
                "B": {"rank": B_snap["rank"], "age": B_snap["age"], "ht": B_snap["ht"], "hand": B_snap["hand"]},
            },
        },
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.models.predict_names 'Player A' 'Player B' [Surface] [BestOf] [Level] [YYYY-MM-DD]")
        sys.exit(1)
    name_a, name_b = sys.argv[1], sys.argv[2]
    surface = sys.argv[3] if len(sys.argv) > 3 else "Hard"
    best_of = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    level   = sys.argv[5] if len(sys.argv) > 5 else "A"
    as_of   = sys.argv[6] if len(sys.argv) > 6 else None
    print(json.dumps(predict_by_names(name_a, name_b, surface, best_of, level, as_of), indent=2))
