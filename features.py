"""Feature engineering for NYK-SAS Finals model.

For a given (team, opponent, date, is_home) tuple, compute a feature vector
using ONLY games that happened before `date`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class FeatureConfig:
    recency_decay: float = 0.95          # weight = decay^(games_ago)
    playoff_weight: float = 3.0          # multiply playoff-game weight
    home_value: float = 3.0              # home court points
    rest_value: float = 0.5              # points per extra rest day
    use_opp_adjustment: bool = True      # subtract opponent strength
    last_n_games: int = 20               # window for rolling stats


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (team_games, league_baseline)."""
    team_games = pd.read_parquet(DATA_DIR / "team_games.parquet")
    team_games["GAME_DATE"] = pd.to_datetime(team_games["GAME_DATE"])
    # League baseline (mean ORtg / DRtg per opponent for opponent adjustment)
    return team_games, None


def league_baselines(team_games: pd.DataFrame) -> dict:
    """Per-team RS averages for opponent adjustment."""
    rs = team_games[team_games["SEASON_TYPE"] == "Regular Season"]
    by_team = rs.groupby("OPP_TEAM_ID").agg(
        opp_off_allowed=("OPP_PTS", "mean"),  # how many points this opponent typically scores
        opp_def_allowed=("PTS", "mean"),  # how many points opponents typically score on them
    )
    league_mean_pts = rs["PTS"].mean()
    return {"by_team": by_team, "league_mean_pts": league_mean_pts}


def team_features(
    team_id: int,
    as_of_date: pd.Timestamp,
    team_games: pd.DataFrame,
    baselines: dict,
    config: FeatureConfig,
) -> dict:
    """Recency-weighted features for `team_id` using games strictly before `as_of_date`."""
    df = team_games[
        (team_games["TEAM_ID"] == team_id) & (team_games["GAME_DATE"] < as_of_date)
    ].sort_values("GAME_DATE", ascending=False)

    df = df.head(config.last_n_games).copy()
    n = len(df)

    # No history → return league-average-ish defaults
    if n == 0:
        return {
            "off_rating": 115.0, "def_rating": 115.0, "net_rating": 0.0,
            "pace": 100.0, "fg_pct": 0.46, "fg3_pct": 0.36,
            "tov": 14.0, "reb": 44.0, "ast": 25.0,
            "momentum_l5": 0.0, "games_played": 0,
        }

    # Weights: recency decay × playoff multiplier
    weights = np.power(config.recency_decay, np.arange(n))
    weights = weights * np.where(df["SEASON_TYPE"].values == "Playoffs", config.playoff_weight, 1.0)
    weights = weights / weights.sum()

    # Opponent-adjusted off / def rating
    off_rating = df["OFF_RATING"].values
    def_rating = df["DEF_RATING"].values
    if config.use_opp_adjustment:
        league_mean = baselines["league_mean_pts"]
        opp_df_def = baselines["by_team"].reindex(df["OPP_TEAM_ID"].values)["opp_def_allowed"].values
        opp_df_off = baselines["by_team"].reindex(df["OPP_TEAM_ID"].values)["opp_off_allowed"].values
        # If opponent had weak defense (allowed many points), inflate our scoring → adjust down
        opp_df_def = np.nan_to_num(opp_df_def, nan=league_mean)
        opp_df_off = np.nan_to_num(opp_df_off, nan=league_mean)
        off_adj = off_rating - (opp_df_def - league_mean)
        def_adj = def_rating - (opp_df_off - league_mean)
    else:
        off_adj, def_adj = off_rating, def_rating

    feats = {
        "off_rating": float(np.average(off_adj, weights=weights)),
        "def_rating": float(np.average(def_adj, weights=weights)),
        "net_rating": float(np.average(df["NET_RATING"].values, weights=weights)),
        "pace": float(np.average(df["PACE"].values, weights=weights)),
        "fg_pct": float(np.average(df["FG_PCT"].values, weights=weights)),
        "fg3_pct": float(np.average(df["FG3_PCT"].values, weights=weights)),
        "tov": float(np.average(df["TOV"].values, weights=weights)),
        "reb": float(np.average(df["REB"].values, weights=weights)),
        "ast": float(np.average(df["AST"].values, weights=weights)),
        # Last 5 momentum
        "momentum_l5": float(df.head(5)["MARGIN"].mean()) if len(df) >= 5 else 0.0,
        "games_played": int(n),
    }
    return feats


def matchup_features(
    home_team_id: int,
    away_team_id: int,
    as_of_date: pd.Timestamp,
    team_games: pd.DataFrame,
    baselines: dict,
    config: FeatureConfig,
    home_rest: int = 1,
    away_rest: int = 1,
) -> dict:
    """Build features for a hypothetical matchup."""
    home = team_features(home_team_id, as_of_date, team_games, baselines, config)
    away = team_features(away_team_id, as_of_date, team_games, baselines, config)

    # NOTE: home_court is NOT a regression feature (it's constant in training,
    # collinear with intercept). It's exposed here so predict_game.py can apply
    # it as a post-hoc adjustment.
    return {
        "off_diff": home["off_rating"] - away["off_rating"],
        "def_diff": away["def_rating"] - home["def_rating"],  # lower opp DEF = better for home
        "net_diff": home["net_rating"] - away["net_rating"],
        "pace_avg": (home["pace"] + away["pace"]) / 2 - 100,  # centered around league avg
        "fg_diff": home["fg_pct"] - away["fg_pct"],
        "fg3_diff": home["fg3_pct"] - away["fg3_pct"],
        "tov_diff": away["tov"] - home["tov"],  # opp tov is good for home
        "reb_diff": home["reb"] - away["reb"],
        "momentum_diff": home["momentum_l5"] - away["momentum_l5"],
        "rest_diff": (home_rest - away_rest) * config.rest_value,
        # Below not used as regression features — exposed for diagnostics
        "_home_court_adjust": config.home_value,
        "home_off": home["off_rating"],
        "home_def": home["def_rating"],
        "away_off": away["off_rating"],
        "away_def": away["def_rating"],
        "home_pace": home["pace"],
        "away_pace": away["pace"],
    }


def perspective_features(
    self_id: int,
    opp_id: int,
    is_home: bool,
    as_of_date: pd.Timestamp,
    team_games: pd.DataFrame,
    baselines: dict,
    config: FeatureConfig,
) -> dict:
    """Compute features from the perspective of `self_id` (their margin vs opp_id).

    All differentials are SELF - OPP. is_home becomes a binary feature.
    """
    s = team_features(self_id, as_of_date, team_games, baselines, config)
    o = team_features(opp_id, as_of_date, team_games, baselines, config)
    return {
        "off_diff": s["off_rating"] - o["off_rating"],
        "def_diff": o["def_rating"] - s["def_rating"],  # lower opp DEF = better for self
        "net_diff": s["net_rating"] - o["net_rating"],
        "pace_avg": (s["pace"] + o["pace"]) / 2 - 100,
        "fg_diff": s["fg_pct"] - o["fg_pct"],
        "fg3_diff": s["fg3_pct"] - o["fg3_pct"],
        "tov_diff": o["tov"] - s["tov"],
        "reb_diff": s["reb"] - o["reb"],
        "momentum_diff": s["momentum_l5"] - o["momentum_l5"],
        "is_home": 1.0 if is_home else 0.0,
    }


def build_training_set(
    team_games: pd.DataFrame,
    config: FeatureConfig,
    playoff_only: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """For every NYK/SAS team-game, compute features from THAT team's perspective.

    Each game appears TWICE in the dataset (once from each team's POV) — but
    we only have NYK and SAS in team_games, so most games appear once.
    Includes IS_HOME as a feature so the model learns home court value properly.

    Targets: MARGIN (self - opp) and TOTAL_POINTS.
    """
    baselines = league_baselines(team_games)

    rows = team_games.copy()
    if playoff_only:
        rows = rows[rows["SEASON_TYPE"] == "Playoffs"]

    feature_rows = []
    margins = []
    totals = []

    for _, g in rows.iterrows():
        feats = perspective_features(
            self_id=g["TEAM_ID"],
            opp_id=g["OPP_TEAM_ID"],
            is_home=bool(g["IS_HOME"]),
            as_of_date=g["GAME_DATE"],
            team_games=team_games,
            baselines=baselines,
            config=config,
        )
        feats["game_id"] = g["GAME_ID"]
        feats["date"] = g["GAME_DATE"]
        feats["self_team"] = g["TEAM_NAME"]
        feats["opp_team"] = g["OPP_TEAM_NAME"]
        feats["season_type"] = g["SEASON_TYPE"]
        feature_rows.append(feats)
        margins.append(g["MARGIN"])
        totals.append(g["TOTAL_POINTS"])

    X = pd.DataFrame(feature_rows)
    y_margin = np.array(margins)
    y_total = np.array(totals)
    return X, y_margin, y_total


if __name__ == "__main__":
    team_games, _ = load_data()
    config = FeatureConfig()
    X, y_margin, y_total = build_training_set(team_games, config)
    print(f"Built training set: {len(X)} games")
    print(f"  Playoff games: {(X['season_type'] == 'Playoffs').sum()}")
    print(f"  Margin mean: {y_margin.mean():.1f}, std: {y_margin.std():.1f}")
    print(f"  Total mean:  {y_total.mean():.1f}, std: {y_total.std():.1f}")
    print(f"\nFeature columns: {[c for c in X.columns if c not in ['game_id','date','home_team','away_team','season_type']]}")
    print(f"\nSample row (most recent game):")
    print(X.iloc[-1].to_dict())
