"""Pull historical NBA Finals data (2015-2025) for calibration.

Outputs:
  - data/historical_finals.parquet — every Finals game from past 11 seasons
                                       (~77 games, all 30 teams covered)

This is the calibration set for:
  - Finals home court value (historical average)
  - Finals pace adjustment (vs regular season)
  - Finals total points distribution
  - Finals margin distribution / blowout rate
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 11 seasons of Finals
SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25",
]


def fetch_season_finals(season: str) -> pd.DataFrame:
    """Fetch all playoff games for a season, then filter to Finals."""
    log = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="T",
        season_type_all_star="Playoffs",
    )
    df = log.get_data_frames()[0]
    if len(df) == 0:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["SEASON"] = season

    # GAME_ID format: "00425XX0YYY" where XX = season code, YY = round, YYY = game
    # Finals = round 4 → game_id contains "004XX004YYY" — extract round digit
    # Actually format: 0042XX0RGGG where R is round-1 (0=R1, 1=R2, 2=ECF/WCF, 3=Finals)
    df["ROUND_CODE"] = df["GAME_ID"].str[7]
    finals = df[df["ROUND_CODE"] == "4"].copy()
    return finals


def join_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Join opponent stats from same GAME_ID."""
    opp_cols = ["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "PTS", "FG_PCT", "FG3_PCT",
                "FGA", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK"]
    opp = df[["GAME_ID"] + opp_cols].copy()
    opp.columns = ["GAME_ID"] + [f"OPP_{c}" for c in opp_cols]
    merged = df.merge(opp, on="GAME_ID")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()
    return merged


def compute_advanced(df: pd.DataFrame) -> pd.DataFrame:
    poss = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    opp_poss = df["OPP_FGA"] + 0.44 * df["OPP_FTA"] - df["OPP_OREB"] + df["OPP_TOV"]
    avg_poss = (poss + opp_poss) / 2

    df = df.copy()
    df["POSSESSIONS"] = avg_poss
    df["PACE"] = 48 * avg_poss / (df["MIN"] / 5)
    df["OFF_RATING"] = 100 * df["PTS"] / avg_poss
    df["DEF_RATING"] = 100 * df["OPP_PTS"] / avg_poss
    df["NET_RATING"] = df["OFF_RATING"] - df["DEF_RATING"]
    df["TOTAL_POINTS"] = df["PTS"] + df["OPP_PTS"]
    df["MARGIN"] = df["PTS"] - df["OPP_PTS"]
    return df


def main():
    print(f"Fetching Finals data for {len(SEASONS)} seasons...")
    all_finals = []
    for season in SEASONS:
        try:
            print(f"  {season}...", end=" ", flush=True)
            df = fetch_season_finals(season)
            print(f"{len(df)} team-game rows ({len(df)//2} games)")
            if len(df) > 0:
                all_finals.append(df)
            time.sleep(0.6)
        except Exception as e:
            print(f"FAILED: {e}")

    if not all_finals:
        print("No data fetched!")
        return

    combined = pd.concat(all_finals, ignore_index=True)
    print(f"\nTotal: {len(combined)} team-game rows, {combined['GAME_ID'].nunique()} games")

    print("Joining opponent stats...")
    combined = join_opponent(combined)
    print(f"After join: {len(combined)} rows")

    print("Computing pace / ratings...")
    combined = compute_advanced(combined)

    combined.to_parquet(DATA_DIR / "historical_finals.parquet", index=False)
    print(f"Saved to {DATA_DIR}/historical_finals.parquet")

    # Summary stats
    print(f"\n{'='*60}")
    print("HISTORICAL FINALS CALIBRATION DATA")
    print('='*60)

    # Home court advantage in Finals
    home = combined[combined["IS_HOME"]]
    away = combined[~combined["IS_HOME"]]
    home_win_pct = (home["WL"] == "W").mean()
    home_avg_margin = home["MARGIN"].mean()
    print(f"Finals home court advantage:")
    print(f"  Home win %: {home_win_pct:.1%} (vs ~57% RS league avg)")
    print(f"  Home avg margin: {home_avg_margin:+.2f} pts")

    # Total points
    games = combined.groupby("GAME_ID").first()
    print(f"\nFinals total points distribution:")
    print(f"  Mean: {games['TOTAL_POINTS'].mean():.1f}")
    print(f"  Std:  {games['TOTAL_POINTS'].std():.1f}")
    print(f"  Min/Max: {games['TOTAL_POINTS'].min()} / {games['TOTAL_POINTS'].max()}")

    # Pace
    print(f"\nFinals pace:")
    print(f"  Mean: {combined['PACE'].mean():.1f}")
    print(f"  Std:  {combined['PACE'].std():.1f}")

    # Margin distribution
    print(f"\nFinals margin distribution (abs):")
    print(f"  Mean: {combined['MARGIN'].abs().mean():.1f}")
    print(f"  Std:  {combined['MARGIN'].std():.1f}")
    print(f"  Blowouts (>15): {(combined['MARGIN'].abs() > 15).mean():.1%}")

    # Per-season summary
    print(f"\nPer-season:")
    by_season = combined.groupby("SEASON").agg(
        games=("GAME_ID", "nunique"),
        avg_total=("TOTAL_POINTS", "mean"),
        avg_pace=("PACE", "mean"),
        home_win_pct=("WL", lambda x: (x == "W").mean()),
    )
    print(by_season.to_string())


if __name__ == "__main__":
    main()
