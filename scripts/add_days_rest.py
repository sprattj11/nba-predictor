# scripts/add_days_rest.py
import pandas as pd
import sys
from pathlib import Path

IN = Path("data/games_summary_merged.csv")
OUT = Path("data/games_summary_merged_with_rest.csv")

if not IN.exists():
    print(f"ERROR: input file not found at {IN}", file=sys.stderr)
    sys.exit(2)

# read and parse date - adjust the date column name if yours differs
df = pd.read_csv(IN, low_memory=False)
date_col = None
for candidate in ("date","game_date","Date"):
    if candidate in df.columns:
        date_col = candidate
        break
if date_col is None:
    raise SystemExit("Could not find a date column (tried date, game_date, Date).")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
if df[date_col].isna().any():
    print("Warning: some dates could not be parsed — check content.", file=sys.stderr)

# keep an original row index to map back
df = df.reset_index().rename(columns={"index":"orig_index"})

# build long-form appearances
if "home_team" not in df.columns or "away_team" not in df.columns:
    raise SystemExit("Could not find 'home_team' or 'away_team' columns.")

home = df[["orig_index", date_col, "home_team"]].rename(columns={date_col: "date", "home_team":"team"})
home["is_home"] = True
away = df[["orig_index", date_col, "away_team"]].rename(columns={date_col: "date", "away_team":"team"})
away["is_home"] = False

long = pd.concat([home, away], ignore_index=True)
long = long.sort_values(["team", "date"])

# compute days since previous game per team
long["days_since_prev"] = long.groupby("team")["date"].diff().dt.days
long["days_since_prev"] = long["days_since_prev"].fillna(3).astype(float)  # default 3 for first appearance

# map back
home_days = long[long["is_home"]].set_index("orig_index")["days_since_prev"]
away_days = long[~long["is_home"]].set_index("orig_index")["days_since_prev"]

df.loc[df["orig_index"].isin(home_days.index), "home_days_rest"] = home_days
df.loc[df["orig_index"].isin(away_days.index), "away_days_rest"] = away_days

# final cleanup and save
df = df.drop(columns=["orig_index"])
df["home_days_rest"] = df["home_days_rest"].fillna(3).astype(float)
df["away_days_rest"] = df["away_days_rest"].fillna(3).astype(float)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
