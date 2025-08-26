# data/convert_atp_tennis_csv.py
import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SERIES2LEVEL = {  # simple map
    "Grand Slam": "G", "Masters 1000": "M",
    "ATP 500": "A", "ATP 250": "A",
    "International": "A", "Team": "D", "Davis Cup": "D",
}

def yyyymmdd(dt):  # to 20250131
    return int(pd.to_datetime(dt).strftime("%Y%m%d"))

def norm(s):  # normalize
    return str(s).lower().replace(".", "").replace("-", " ").replace("'", "").strip()

def split_abbrev(name):  # "Alcaraz C." -> ("alcaraz","c")
    name = str(name).strip()
    if "," in name:
        last, first = [t.strip() for t in name.split(",", 1)]
        return norm(last), norm(first)[:1]
    parts = name.split()
    if len(parts) == 1:
        return norm(parts[0]), ""
    last = " ".join(parts[:-1])
    fi = parts[-1][:1]
    return norm(last), norm(fi)

def load_players():
    p = RAW / "atp_players.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    df = pd.read_csv(
        p,
        dtype={
            "player_id": "Int64",
            "name_first": "string",
            "name_last": "string",
            "hand": "string",
            "dob": "string",           # YYYYMMDD as string
            "ioc": "string",
            "height": "float64",
            "wikidata_id": "string",   # avoids DtypeWarning
        },
        low_memory=False,
    )
    df["last_n"] = df["name_last"].map(norm)
    df["first_initial"] = df["name_first"].fillna("").map(lambda x: str(x).strip().lower()[:1])
    # keep first per (last_n, first_initial)
    idx = (df.sort_values(["player_id"])
             .drop_duplicates(subset=["last_n","first_initial"], keep="first")
             .reset_index(drop=True))
    return df, idx

def lookup_player(row_name, players_idx):
    last_n, fi = split_abbrev(row_name)
    hit = players_idx[(players_idx["last_n"] == last_n) &
                      ((players_idx["first_initial"] == fi) | (fi == ""))]
    if len(hit):
        r = hit.iloc[0]
        full = f"{str(r['name_first']).strip()} {str(r['name_last']).strip()}".strip()
        return int(r["player_id"]), full, r
    return -1, str(row_name), None  # temp id

def age_on(date_yyyymmdd, dob_str):  # dob_str like "20030505"
    if pd.isna(dob_str) or str(dob_str).strip() == "":
        return np.nan
    d0 = datetime.strptime(str(date_yyyymmdd), "%Y%m%d").date()
    d1 = pd.to_datetime(str(dob_str), format="%Y%m%d", errors="coerce")
    if pd.isna(d1):  # fallback if bad
        return np.nan
    return (d0 - d1.date()).days / 365.25

def convert():
    players_all, players_idx = load_players()

    src = RAW / "atp_tennis.csv"  # tennis-data.co.uk merged file
    if not src.exists():
        raise FileNotFoundError(f"missing {src}")
    raw = pd.read_csv(src, low_memory=False)

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw[raw["Date"].dt.year >= 2025].copy()
    if raw.empty:
        raise ValueError("no 2025 rows in atp_tennis.csv")

    # ensure ranks numeric; map "NR" -> NaN
    for c in ["Rank_1","Rank_2"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    out_rows = []
    for _, r in raw.iterrows():
        date_i = yyyymmdd(r["Date"])
        level = SERIES2LEVEL.get(str(r.get("Series")), "A")
        best_of = int(r.get("Best of", 3) if pd.notna(r.get("Best of")) else 3)

        p1, p2 = r["Player_1"], r["Player_2"]
        wname = r["Winner"]
        lname = p2 if wname == p1 else p1

        w_rank = r["Rank_1"] if wname == p1 else r["Rank_2"]
        l_rank = r["Rank_2"] if wname == p1 else r["Rank_1"]

        w_id, w_full, w_row = lookup_player(wname, players_idx)
        l_id, l_full, l_row = lookup_player(lname, players_idx)

        w_hand = (w_row["hand"] if w_row is not None else np.nan)
        l_hand = (l_row["hand"] if l_row is not None else np.nan)
        w_ht   = (w_row["height"] if w_row is not None else np.nan)
        l_ht   = (l_row["height"] if l_row is not None else np.nan)
        w_age  = age_on(date_i, w_row["dob"]) if w_row is not None else np.nan
        l_age  = age_on(date_i, l_row["dob"]) if l_row is not None else np.nan

        out_rows.append({
            "tourney_date": date_i,
            "surface": r.get("Surface"),
            "tourney_level": level,
            "best_of": best_of,
            "winner_name": w_full,
            "loser_name": l_full,
            "winner_id": w_id,
            "loser_id": l_id,
            "winner_rank": int(w_rank) if pd.notna(w_rank) else np.nan,
            "loser_rank": int(l_rank) if pd.notna(l_rank) else np.nan,
            "winner_age": w_age,
            "winner_ht": w_ht,
            "winner_hand": w_hand,
            "loser_age": l_age,
            "loser_ht": l_ht,
            "loser_hand": l_hand,
            "round": r.get("Round", ""),
        })

    out = pd.DataFrame(out_rows).sort_values("tourney_date").reset_index(drop=True)
    dst = RAW / "atp_matches_2025.csv"
    out.to_csv(dst, index=False)

    n_tmp = int((out["winner_id"] < 0).sum() + (out["loser_id"] < 0).sum())
    print(f"wrote {len(out)} matches -> {dst}")
    print(f"unmapped players (temp ids): {n_tmp}")

if __name__ == "__main__":
    convert()
