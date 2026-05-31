"""Monte Carlo simulation of full NBA Finals series.

Uses per-game win probabilities (with home/away alternation following the
2-2-1-1-1 pattern) and simulates 10,000 series to estimate:
  - P(NYK wins series in 4, 5, 6, 7)
  - P(SAS wins series in 4, 5, 6, 7)
  - Expected series length
  - Most likely outcome
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from features import (FeatureConfig, build_training_set, load_data,
                       perspective_features, league_baselines,
                       team_features as tf)
from model import BayesianLinearModel
from predict_game import load_best_config, analytic_prediction, TEAM_IDS, HOME_COURT_FINALS, MARGIN_STD


# 2-2-1-1-1 pattern (better seed has home court for games 1, 2, 5, 7)
# SAS is higher seed (62-20 vs 53-29), so they're home for 1, 2, 5, 7
FINALS_HOME_PATTERN = {
    1: "SAS", 2: "SAS",
    3: "NYK", 4: "NYK", 5: "NYK" if False else "SAS",  # game 5 returns to higher seed
    5: "SAS", 6: "NYK", 7: "SAS",
}


_MODEL_CACHE: dict = {}


def _get_models(as_of_date: pd.Timestamp):
    """Cache: train Bayesian models once per (date, config)."""
    key = str(as_of_date.date())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    config, feature_cols, prior_prec, _ = load_best_config()
    team_games, _ = load_data()
    baselines = league_baselines(team_games)

    X_df, y_margin, y_total = build_training_set(team_games, config)
    is_playoff = (X_df["season_type"] == "Playoffs").values
    sw = np.where(is_playoff, config.playoff_weight, 1.0)
    margin_model = BayesianLinearModel(prior_precision=prior_prec)
    margin_model.fit(X_df[feature_cols].values, y_margin, sample_weight=sw)
    total_model = BayesianLinearModel(prior_precision=prior_prec)
    total_model.fit(X_df[feature_cols].values, y_total, sample_weight=sw)

    cached = {
        "config": config,
        "feature_cols": feature_cols,
        "margin_model": margin_model,
        "total_model": total_model,
        "team_games": team_games,
        "baselines": baselines,
    }
    _MODEL_CACHE[key] = cached
    return cached


def per_game_winprob(home: str, away: str, as_of_date: pd.Timestamp, weight_analytic: float = 0.7) -> dict:
    """Returns win prob for `home` plus margin/total predictions, blending analytic + bayes."""
    cached = _get_models(as_of_date)
    config = cached["config"]
    feature_cols = cached["feature_cols"]
    team_games = cached["team_games"]
    baselines = cached["baselines"]

    home_id = TEAM_IDS[home]
    away_id = TEAM_IDS[away]
    h_feats = tf(home_id, as_of_date, team_games, baselines, config)
    a_feats = tf(away_id, as_of_date, team_games, baselines, config)

    # Analytic
    analytic = analytic_prediction(h_feats, a_feats)

    # Bayesian (cached models)
    margin_model = cached["margin_model"]
    total_model = cached["total_model"]
    feats = perspective_features(home_id, away_id, is_home=True,
                                  as_of_date=as_of_date,
                                  team_games=team_games,
                                  baselines=baselines, config=config)
    x_pred = np.array([[feats[c] for c in feature_cols]])
    b_margin = float(margin_model.predict_distribution(x_pred)["mean"][0])
    b_total = float(total_model.predict_distribution(x_pred)["mean"][0])
    b_winp = float(margin_model.win_prob(x_pred)[0])

    # Blend: weight_analytic on analytic, rest on bayes
    blended_margin = weight_analytic * analytic["margin"] + (1 - weight_analytic) * b_margin
    blended_total = weight_analytic * analytic["total"] + (1 - weight_analytic) * b_total
    blended_winp = float(norm.cdf(blended_margin / MARGIN_STD))

    return {
        "home": home, "away": away,
        "analytic": analytic,
        "bayesian": {"margin": b_margin, "total": b_total, "win_prob": b_winp},
        "blended": {
            "margin": blended_margin,
            "total": blended_total,
            "win_prob": blended_winp,
        },
        "home_form": h_feats,
        "away_form": a_feats,
    }


def simulate_series(per_game_probs: list[float], n_sims: int = 10000, seed: int = 42) -> dict:
    """Simulate `n_sims` 7-game series.

    per_game_probs[i] = probability NYK wins game i+1
    Returns counts/probs for each outcome (NYK in 4/5/6/7, SAS in 4/5/6/7).
    """
    rng = np.random.default_rng(seed)
    outcomes = {f"NYK in {n}": 0 for n in range(4, 8)}
    outcomes.update({f"SAS in {n}": 0 for n in range(4, 8)})
    nyk_wins_total = 0

    for _ in range(n_sims):
        nyk_wins = 0
        sas_wins = 0
        games_played = 0
        for i in range(7):
            games_played += 1
            if rng.random() < per_game_probs[i]:
                nyk_wins += 1
            else:
                sas_wins += 1
            if nyk_wins == 4:
                outcomes[f"NYK in {games_played}"] += 1
                nyk_wins_total += 1
                break
            if sas_wins == 4:
                outcomes[f"SAS in {games_played}"] += 1
                break

    return {
        "n_sims": n_sims,
        "outcomes": {k: v / n_sims for k, v in outcomes.items()},
        "counts": outcomes,
        "p_nyk_wins": nyk_wins_total / n_sims,
        "p_sas_wins": 1 - nyk_wins_total / n_sims,
        "expected_games": float(sum(
            int(k.split()[-1]) * (v / n_sims) for k, v in outcomes.items()
        )),
    }


def run_full_sim(out_path: str | None = None, n_sims: int = 10000) -> dict:
    """Compute per-game win probabilities for all 7 games and simulate the series."""
    print("Building per-game predictions...")

    # Higher seed (SAS) has home for games 1, 2, 5, 7
    home_pattern = {1: "SAS", 2: "SAS", 3: "NYK", 4: "NYK", 5: "SAS", 6: "NYK", 7: "SAS"}

    as_of = pd.Timestamp("2026-06-04")
    per_game = []
    nyk_per_game_probs = []

    for game_num in range(1, 8):
        home = home_pattern[game_num]
        away = "NYK" if home == "SAS" else "SAS"
        pred = per_game_winprob(home, away, as_of)
        nyk_wins_this_game = (1 - pred["blended"]["win_prob"]) if home == "SAS" else pred["blended"]["win_prob"]
        nyk_per_game_probs.append(nyk_wins_this_game)
        per_game.append({
            "game": game_num,
            "home": home,
            "away": away,
            "home_win_prob": pred["blended"]["win_prob"],
            "nyk_win_prob": nyk_wins_this_game,
            "predicted_margin_home": pred["blended"]["margin"],
            "predicted_total": pred["blended"]["total"],
            "analytic_margin_home": pred["analytic"]["margin"],
            "bayes_margin_home": pred["bayesian"]["margin"],
        })
        print(f"  Game {game_num}: {away} @ {home}  →  NYK wins {nyk_wins_this_game:.1%}  "
              f"(margin home {pred['blended']['margin']:+.1f}, total {pred['blended']['total']:.1f})")

    print(f"\nSimulating series ({n_sims:,} trials)...")
    sim = simulate_series(nyk_per_game_probs, n_sims=n_sims)

    print(f"\n{'Outcome':<14} {'Probability':<12}")
    print("-" * 30)
    for outcome, prob in sorted(sim["outcomes"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 50)
        print(f"  {outcome:<11} {prob:6.1%}  {bar}")
    print(f"\nP(NYK wins series): {sim['p_nyk_wins']:.1%}")
    print(f"P(SAS wins series): {sim['p_sas_wins']:.1%}")
    print(f"Expected games: {sim['expected_games']:.1f}")

    result = {
        "per_game": per_game,
        "simulation": sim,
        "home_pattern": home_pattern,
        "nyk_per_game_probs": nyk_per_game_probs,
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")

    return result


if __name__ == "__main__":
    run_full_sim(out_path="logs/series_simulation.json")
