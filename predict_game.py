"""Predict a Finals game.

Two models report:
  1. BAYESIAN: linear regression with autoresearch-selected features
  2. ANALYTIC: simple power-rating formula
        predicted_margin = HOME_COURT + (home_net_rating - away_net_rating)

The analytic model uses domain knowledge (a 1pt net-rating advantage ≈ 1pt
margin) which the Bayesian regression can't reliably learn from 16 playoff
games. We report both and take the consensus.

Usage:
    uv run python predict_game.py --home SAS --away NYK
    uv run python predict_game.py --home NYK --away SAS  --game 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from features import (FeatureConfig, build_training_set, load_data,
                       perspective_features, league_baselines,
                       team_features as tf)
from model import BayesianLinearModel

MODEL_DIR = Path(__file__).parent / "model"
LOG_DIR = Path(__file__).parent / "logs"

TEAM_IDS = {
    "NYK": 1610612752,
    "SAS": 1610612759,
}

# Calibrated from 11 seasons of Finals data (2014-15 to 2024-25, 63 games)
HOME_COURT_FINALS = 4.57    # historical avg home margin in Finals
MARGIN_STD = 14.3            # historical Finals margin std
FINALS_AVG_TOTAL = 210.9     # historical Finals total points mean
FINALS_TOTAL_STD = 17.0      # historical Finals total points std
FINALS_AVG_PACE = 97.3       # historical Finals pace
FINALS_TOTAL_SHRINKAGE = 0.5 # weight on historical avg vs model raw prediction


def load_best_config():
    with open(MODEL_DIR / "best_config.json") as f:
        best = json.load(f)
    config = FeatureConfig(
        recency_decay=float(best["recency_decay"]),
        playoff_weight=float(best["playoff_weight"]),
        home_value=float(best["home_value"]),
        rest_value=float(best["rest_value"]),
        use_opp_adjustment=bool(best["use_opp_adjustment"]),
        last_n_games=int(best["last_n_games"]),
    )
    feature_cols = best["feature_cols"].split(",")
    return config, feature_cols, float(best.get("prior_precision", 1.0)), best


def analytic_prediction(home_feats: dict, away_feats: dict) -> dict:
    """Canonical power-rating formula, calibrated with historical Finals data.

    Margin: home court advantage + net rating differential
    Total: blend of (raw scoring estimate × Finals-adjusted pace) and
           historical Finals average — shrunk toward 210.9 because models
           trained on RS data systematically over-predict totals in Finals.
    """
    net_diff = home_feats["net_rating"] - away_feats["net_rating"]
    margin = HOME_COURT_FINALS + net_diff

    # Pace: blend team pace with Finals avg pace (defenses tighten in Finals)
    raw_pace = (home_feats["pace"] + away_feats["pace"]) / 2
    finals_pace = 0.5 * raw_pace + 0.5 * FINALS_AVG_PACE

    # Total: naive estimate, then shrink toward historical Finals mean
    avg_ortg = (home_feats["off_rating"] + away_feats["off_rating"]) / 2
    raw_total = avg_ortg * finals_pace / 100 * 2
    total = FINALS_TOTAL_SHRINKAGE * FINALS_AVG_TOTAL + (1 - FINALS_TOTAL_SHRINKAGE) * raw_total

    return {
        "margin": margin,
        "total": total,
        "raw_total": raw_total,
        "win_prob": float(norm.cdf(margin / MARGIN_STD)),
        "net_diff": net_diff,
    }


def predict(home: str, away: str, as_of_date: pd.Timestamp | None = None, game_num: int = 1):
    config, feature_cols, prior_prec, best_cfg = load_best_config()
    team_games, _ = load_data()
    if as_of_date is None:
        as_of_date = pd.Timestamp("2026-06-04")  # Finals Game 1

    baselines = league_baselines(team_games)
    home_id, away_id = TEAM_IDS[home], TEAM_IDS[away]
    h_feats = tf(home_id, as_of_date, team_games, baselines, config)
    a_feats = tf(away_id, as_of_date, team_games, baselines, config)

    # === ANALYTIC MODEL ===
    analytic = analytic_prediction(h_feats, a_feats)

    # === BAYESIAN MODEL ===
    X_df, y_margin, y_total = build_training_set(team_games, config)
    is_playoff = (X_df["season_type"] == "Playoffs").values
    sample_weights = np.where(is_playoff, config.playoff_weight, 1.0)
    X = X_df[feature_cols].values

    margin_model = BayesianLinearModel(prior_precision=prior_prec)
    margin_model.fit(X, y_margin, sample_weight=sample_weights)
    total_model = BayesianLinearModel(prior_precision=prior_prec)
    total_model.fit(X, y_total, sample_weight=sample_weights)

    feats = perspective_features(home_id, away_id, is_home=True,
                                  as_of_date=as_of_date,
                                  team_games=team_games,
                                  baselines=baselines, config=config)
    x_pred = np.array([[feats[c] for c in feature_cols]])
    bayes_margin = margin_model.predict_distribution(x_pred)
    bayes_total = total_model.predict_distribution(x_pred)
    bayes_winp = float(margin_model.win_prob(x_pred)[0])

    # === CONSENSUS ===
    consensus_margin = (analytic["margin"] + bayes_margin["mean"][0]) / 2
    consensus_total = (analytic["total"] + bayes_total["mean"][0]) / 2
    consensus_winp = float(norm.cdf(consensus_margin / MARGIN_STD))

    # === PRINT ===
    print("=" * 72)
    print(f"FINALS GAME {game_num}: {away} @ {home}  ({as_of_date.date()})")
    print("=" * 72)

    print(f"\nRecent form (weighted by recency × playoff multiplier):")
    print(f"  {home}: ORtg {h_feats['off_rating']:6.1f}  DRtg {h_feats['def_rating']:6.1f}  "
          f"NetRtg {h_feats['net_rating']:+6.1f}  Pace {h_feats['pace']:5.1f}  "
          f"L5 Margin {h_feats['momentum_l5']:+5.1f}")
    print(f"  {away}: ORtg {a_feats['off_rating']:6.1f}  DRtg {a_feats['def_rating']:6.1f}  "
          f"NetRtg {a_feats['net_rating']:+6.1f}  Pace {a_feats['pace']:5.1f}  "
          f"L5 Margin {a_feats['momentum_l5']:+5.1f}")

    print(f"\n{'Model':<14} {'Home WinP':<12} {'Margin':<14} {'Total':<10}")
    print("-" * 72)
    print(f"{'Analytic':<14} {analytic['win_prob']:<12.1%} "
          f"{analytic['margin']:+.1f}{'':<10} {analytic['total']:<10.1f}")
    print(f"{'Bayesian':<14} {bayes_winp:<12.1%} "
          f"{bayes_margin['mean'][0]:+.1f} (±{bayes_margin['std'][0]:.1f}){'':<3} "
          f"{bayes_total['mean'][0]:<5.1f} (±{bayes_total['std'][0]:.1f})")
    print(f"{'CONSENSUS':<14} {consensus_winp:<12.1%} "
          f"{consensus_margin:+.1f}{'':<10} {consensus_total:<10.1f}")

    # Save
    out = {
        "as_of_date": str(as_of_date.date()),
        "home_team": home,
        "away_team": away,
        "game_num": game_num,
        "analytic": analytic,
        "bayesian": {
            "win_prob": bayes_winp,
            "margin_mean": float(bayes_margin["mean"][0]),
            "margin_std": float(bayes_margin["std"][0]),
            "margin_80ci": [float(bayes_margin["lo_10"][0]), float(bayes_margin["hi_90"][0])],
            "total_mean": float(bayes_total["mean"][0]),
            "total_std": float(bayes_total["std"][0]),
        },
        "consensus": {
            "win_prob": consensus_winp,
            "margin": consensus_margin,
            "total": consensus_total,
        },
        "home_form": h_feats,
        "away_form": a_feats,
    }
    out_path = LOG_DIR / f"prediction_g{game_num}_{home}_vs_{away}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--home", required=True, choices=list(TEAM_IDS.keys()))
    p.add_argument("--away", required=True, choices=list(TEAM_IDS.keys()))
    p.add_argument("--date", help="YYYY-MM-DD")
    p.add_argument("--game", type=int, default=1)
    args = p.parse_args()
    as_of = pd.Timestamp(args.date) if args.date else None
    predict(args.home, args.away, as_of, args.game)
