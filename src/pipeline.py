import json
import os

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "nba_stats" / "Advanced.csv"
OUT_PATH = ROOT / "data" / "processed" / "cleaned_nba.csv"
LOG_PATH = ROOT / "data" / "processed" / "log.json"

FEATURES=["usg_percent","ts_percent","x3p_ar","ast_percent","trb_percent"]

ID_COLS=["player","season","tm",'mp']
OPTIONAL_COLS=["pos"]
FIRST_SEASON,LAST_SEASON=2016,2024
MIN_MINUTES=500

TRAIN_END, VAL_END=2021,2023

COMBINED_ROWS={"TOT","2TM","3TM","4TM","5TM"}

def assign_split(season):
    if season <= TRAIN_END:
        return "train"
    elif season <= VAL_END:
        return "val"
    else:
        return "test"
def load_clean(path=RAW_PATH):
    adv = pd.read_csv(path)
    cols = ID_COLS + FEATURES
    missing = [c for c in cols if c not in adv.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}\nAvailable: {list(adv.columns)}")
    keep = cols + [c for c in OPTIONAL_COLS if c in adv.columns]

    log = {"raw_rows": len(adv)}
    df = adv[keep].copy()
    for c in ["season", "mp"] + FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

   
    n = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = n - len(df)
    df = df[df["season"].between(FIRST_SEASON, LAST_SEASON)]
    log["rows_in_season_range"] = len(df)
    n = len(df)
    df = df.dropna(subset=["season", "mp"] + FEATURES)
    log["rows_dropped_missing"] = n - len(df)

    is_combined = df["tm"].astype(str).str.upper().isin(COMBINED_ROWS)
    keys = set(zip(df.loc[is_combined, "player"], df.loc[is_combined, "season"]))
    in_traded = pd.Series([(p, s) in keys for p, s in zip(df["player"], df["season"])], index=df.index)
    log["combined_rows_found"] = int(is_combined.sum())
    log["per_team_rows_dropped"] = int((in_traded & ~is_combined).sum())
    df = df[~in_traded | is_combined]

    n = len(df)
    df = df[df["mp"] >= MIN_MINUTES].reset_index(drop=True)
    log["removed_by_minutes_filter"] = n - len(df)

    df["split"] = df["season"].astype(int).map(assign_split)
    log["final_rows"] = len(df)
    log["rows_per_split"] = df["split"].value_counts().to_dict()
    return df, log


def main():
    df, log = load_clean()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()