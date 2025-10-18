#!/usr/bin/env python3
"""
Merge game-level metadata (date and true home/away) into games_summary.csv
Requires a games-level file (e.g., data/games.csv) that contains GAME_ID and
home/visitor team identifiers and optionally GAME_DATE_EST.

Usage:
python models/merge_game_metadata.py \
  --summary data/games_summary.csv \
  --games data/games.csv \
  --out data/games_summary_merged.csv
"""
import argparse
from pathlib import Path
import pandas as pd

def find_id_like(cols):
    for c in cols:
        if c.upper() in ("GAME_ID","GAMEID","GAME_ID_x","GAME_ID_y"):
            return c
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--games", required=True)
    p.add_argument("--out", default="data/games_summary_merged.csv")
    args = p.parse_args()

    s = pd.read_csv(args.summary, low_memory=False)
    g = pd.read_csv(args.games, low_memory=False)

    # identify GAME_ID column names
    sid = find_id_like(s.columns)
    gid = find_id_like(g.columns)
    if sid is None or gid is None:
        raise SystemExit("Couldn't detect GAME_ID column in one of the files. Check headers and re-run.")

    # canonicalize column name to GAME_ID for merging
    s = s.rename(columns={sid: "GAME_ID"})
    g = g.rename(columns={gid: "GAME_ID"})

    # Attempt to find date and home/visitor in games file
    date_col = next((c for c in g.columns if "date" in c.lower()), None)
    home_col = next((c for c in g.columns if "home" in c.lower() and ("team" in c.lower() or "id" in c.lower() or "abbr" in c.lower())), None)
    vis_col  = next((c for c in g.columns if ("visitor" in c.lower() or "away" in c.lower()) and ("team" in c.lower() or "id" in c.lower() or "abbr" in c.lower())), None)

    selected = ["GAME_ID"]
    if date_col:
        selected.append(date_col)
    if home_col:
        selected.append(home_col)
    if vis_col:
        selected.append(vis_col)

    meta = g[selected].drop_duplicates(subset="GAME_ID")
    out = s.merge(meta, on="GAME_ID", how="left", suffixes=("","_meta"))

    # normalize column names
    if date_col:
        out = out.rename(columns={date_col: "date"})
    if home_col:
        out = out.rename(columns={home_col: "home_team_meta"})
    if vis_col:
        out = out.rename(columns={vis_col: "away_team_meta"})

    # If the summary already had home/away cols, prefer them. Otherwise use meta.
    if "home_team" not in out.columns and "home_team_meta" in out.columns:
        out = out.rename(columns={"home_team_meta":"home_team"})
    if "away_team" not in out.columns and "away_team_meta" in out.columns:
        out = out.rename(columns={"away_team_meta":"away_team"})

    # coerce date to datetime if present
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out.to_csv(args.out, index=False)
    print("Wrote merged summary to", args.out)
    print("Columns now:", list(out.columns))

if __name__ == "__main__":
    main()
