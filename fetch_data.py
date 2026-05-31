"""Pull all data needed for the NYK-SAS Finals model.

Outputs to data/:
  - team_games.parquet      every NYK/SAS team-game (RS + playoffs) with opponent stats joined
  - playoff_games.parquet   playoff-only subset
  - h2h.parquet             the 4 NYK-SAS regular-season meetings
"""

from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams as nba_teams

SEASON = "2025-26"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TEAMS = {t["full_name"]: t["id"] for t in nba_teams.get_teams()}
NYK_ID = TEAMS["New York Knicks"]
SAS_ID = TEAMS["San Antonio Spurs"]


def fetch_league_log(season_type: str) -> pd.DataFrame:
    log = leaguegamelog.LeagueGameLog(
        season=SEASON,
        player_or_team_abbreviation="T",
        season_type_all_star=season_type,
    )
    df = log.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["SEASON_TYPE"] = season_type
    return df


def join_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, attach the opponent team's stats from the same GAME_ID."""
    opp_cols = ["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "PTS", "FG_PCT", "FG3_PCT",
                "FGA", "FTA", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK"]
    opp = df[["GAME_ID"] + opp_cols].copy()
    opp.columns = ["GAME_ID"] + [f"OPP_{c}" for c in opp_cols]

    merged = df.merge(opp, on="GAME_ID")
    # remove self-rows
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()
    return merged


def compute_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pace, off-rating, def-rating per team-game (Dean Oliver approximation)."""
    poss = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    opp_poss = df["OPP_FGA"] + 0.44 * df["OPP_FTA"] - df["OPP_OREB"] + df["OPP_TOV"]
    avg_poss = (poss + opp_poss) / 2

    df = df.copy()
    df["POSSESSIONS"] = avg_poss
    df["PACE"] = 48 * avg_poss / (df["MIN"] / 5)  # MIN is total team minutes (5 players)
    df["OFF_RATING"] = 100 * df["PTS"] / avg_poss
    df["DEF_RATING"] = 100 * df["OPP_PTS"] / avg_poss
    df["NET_RATING"] = df["OFF_RATING"] - df["DEF_RATING"]
    df["TOTAL_POINTS"] = df["PTS"] + df["OPP_PTS"]
    df["MARGIN"] = df["PTS"] - df["OPP_PTS"]
    return df


def main():
    print(f"Fetching {SEASON} regular season log...")
    rs = fetch_league_log("Regular Season")
    print(f"  {len(rs)} team-game rows")

    print(f"Fetching {SEASON} playoff log...")
    po = fetch_league_log("Playoffs")
    print(f"  {len(po)} team-game rows")

    all_games = pd.concat([rs, po], ignore_index=True)

    # Need opponent join BEFORE filtering to NYK/SAS, otherwise we lose opponent rows
    print("Joining opponent stats...")
    all_joined = join_opponent(all_games)
    print(f"  {len(all_joined)} rows after opponent join")

    # Compute advanced stats
    print("Computing pace / ratings...")
    all_advanced = compute_advanced(all_joined)

    # Filter to NYK + SAS
    target = all_advanced[all_advanced["TEAM_ID"].isin([NYK_ID, SAS_ID])].copy()
    target = target.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    nyk = target[target["TEAM_ID"] == NYK_ID]
    sas = target[target["TEAM_ID"] == SAS_ID]
    print(f"\n  NYK: {len(nyk)} games ({(nyk['SEASON_TYPE'] == 'Playoffs').sum()} playoff)")
    print(f"  SAS: {len(sas)} games ({(sas['SEASON_TYPE'] == 'Playoffs').sum()} playoff)")

    target.to_parquet(DATA_DIR / "team_games.parquet", index=False)

    # Playoff-only
    po_only = target[target["SEASON_TYPE"] == "Playoffs"]
    po_only.to_parquet(DATA_DIR / "playoff_games.parquet", index=False)

    # H2H (both teams in same game)
    h2h_ids = set(nyk["GAME_ID"]) & set(sas["GAME_ID"])
    h2h = target[target["GAME_ID"].isin(h2h_ids)].copy()
    h2h.to_parquet(DATA_DIR / "h2h.parquet", index=False)
    print(f"  H2H games: {len(h2h_ids)}")

    print("\nDone. Files written to data/")
    for f in sorted(DATA_DIR.glob("*.parquet")):
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")

    print("\nSummary stats:")
    for team_name, team_df in [("NYK", nyk), ("SAS", sas)]:
        po_df = team_df[team_df["SEASON_TYPE"] == "Playoffs"]
        rs_df = team_df[team_df["SEASON_TYPE"] == "Regular Season"]
        print(f"  {team_name} playoff: ORtg={po_df['OFF_RATING'].mean():.1f}  "
              f"DRtg={po_df['DEF_RATING'].mean():.1f}  Pace={po_df['PACE'].mean():.1f}  "
              f"Total={po_df['TOTAL_POINTS'].mean():.1f}")
        print(f"  {team_name} RS:      ORtg={rs_df['OFF_RATING'].mean():.1f}  "
              f"DRtg={rs_df['DEF_RATING'].mean():.1f}  Pace={rs_df['PACE'].mean():.1f}  "
              f"Total={rs_df['TOTAL_POINTS'].mean():.1f}")


if __name__ == "__main__":
    main()
