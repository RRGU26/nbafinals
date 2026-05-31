"""Karpathy-style autoresearch loop for the Finals model.

Search over:
  - Recency decay rate
  - Playoff sample weight
  - Home court value
  - Feature subsets

Scoring: leave-one-out cross-validation on PLAYOFF GAMES only (the regime we
care about). Score = negative RMSE on margin + 0.5 * negative RMSE on total.

Outputs:
  - logs/experiments.parquet: every experiment's config + score
  - model/best_config.json: best config + selected features
"""

import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from features import FeatureConfig, build_training_set, load_data
from model import BayesianLinearModel

LOG_DIR = Path(__file__).parent / "logs"
MODEL_DIR = Path(__file__).parent / "model"
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

ALL_FEATURES = [
    "off_diff", "def_diff", "net_diff", "pace_avg",
    "fg_diff", "fg3_diff", "tov_diff", "reb_diff",
    "momentum_diff", "rest_diff",
]


def feature_subsets() -> list[list[str]]:
    """Generate candidate feature sets.

    Always include is_home (so model learns home court value).
    Then add 1-2 strength signals on top.
    """
    subsets = []
    base = ["is_home"]
    primary = [["net_diff"], ["off_diff", "def_diff"]]
    secondary = ["fg_diff", "fg3_diff", "tov_diff", "reb_diff",
                 "momentum_diff", "pace_avg"]
    for p in primary:
        subsets.append(base + p)
        for s in secondary:
            subsets.append(base + p + [s])
    return subsets


def loo_cv_score(X: np.ndarray, y: np.ndarray, weights: np.ndarray, target_mask: np.ndarray, prior_precision: float = 1.0) -> tuple[float, list, list]:
    """Leave-one-out CV RMSE, only scoring on rows where target_mask is True."""
    n = len(y)
    errors = []
    preds = []
    actuals = []
    for i in range(n):
        if not target_mask[i]:
            continue
        train_idx = np.arange(n) != i
        model = BayesianLinearModel(prior_precision=prior_precision)
        model.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
        pred, _ = model.predict(X[i:i+1])
        errors.append((pred[0] - y[i]) ** 2)
        preds.append(pred[0])
        actuals.append(y[i])
    if not errors:
        return float("inf"), [], []
    return float(np.sqrt(np.mean(errors))), preds, actuals


def run_experiment(
    config: FeatureConfig,
    feature_cols: list[str],
    team_games: pd.DataFrame,
    prior_precision: float = 1.0,
) -> dict:
    X_df, y_margin, y_total = build_training_set(team_games, config)
    is_playoff = (X_df["season_type"] == "Playoffs").values

    X = X_df[feature_cols].values
    sample_weights = np.where(is_playoff, config.playoff_weight, 1.0)

    # Sign sanity check on full-data fit: coefficients should have correct signs
    sanity_model = BayesianLinearModel(prior_precision=prior_precision)
    sanity_model.fit(X, y_margin, sample_weight=sample_weights)
    native_coefs = sanity_model.coefficients_native_scale(feature_cols)

    expected_signs = {
        "off_diff": +1, "def_diff": +1, "net_diff": +1,
        "fg_diff": +1, "fg3_diff": +1, "reb_diff": +1,
        "tov_diff": +1, "momentum_diff": +1, "pace_avg": 0,  # pace can go either way
        "is_home": +1,
    }
    sign_violations = 0
    for f in feature_cols:
        if f in expected_signs and expected_signs[f] != 0:
            if native_coefs[f] * expected_signs[f] < 0:
                sign_violations += 1

    margin_rmse, margin_preds, margin_actuals = loo_cv_score(X, y_margin, sample_weights, target_mask=is_playoff, prior_precision=prior_precision)
    total_rmse, _, _ = loo_cv_score(X, y_total, sample_weights, target_mask=is_playoff, prior_precision=prior_precision)

    # Win-direction accuracy
    if margin_preds:
        correct = sum(1 for p, a in zip(margin_preds, margin_actuals) if (p > 0) == (a > 0))
        win_acc = correct / len(margin_preds)
    else:
        win_acc = 0.0

    return {
        "recency_decay": config.recency_decay,
        "playoff_weight": config.playoff_weight,
        "home_value": config.home_value,
        "rest_value": config.rest_value,
        "use_opp_adjustment": config.use_opp_adjustment,
        "last_n_games": config.last_n_games,
        "prior_precision": prior_precision,
        "feature_cols": ",".join(feature_cols),
        "n_features": len(feature_cols),
        "margin_rmse": margin_rmse,
        "total_rmse": total_rmse,
        "win_accuracy": win_acc,
        "sign_violations": sign_violations,
        # HARD penalty for wrong-signed coefficients (they won't generalize)
        "combined_score": margin_rmse + 0.5 * total_rmse - 10 * win_acc + 100 * sign_violations,
    }


def main(max_experiments: int = 200):
    team_games, _ = load_data()
    print(f"Loaded {len(team_games)} team-games")

    # Build search grid — much stronger priors to combat overfitting
    configs = []
    for decay in [0.90, 0.95, 0.97, 1.0]:
        for pw in [1.0, 2.0, 3.0, 5.0]:
            for opp_adj in [True, False]:
                for n_games in [20, 30, 50]:
                    for prior in [3.0, 10.0, 30.0, 100.0]:
                        c = FeatureConfig(
                            recency_decay=decay,
                            playoff_weight=pw,
                            use_opp_adjustment=opp_adj,
                            last_n_games=n_games,
                        )
                        configs.append((c, prior))
    subsets = feature_subsets()
    print(f"Configs: {len(configs)}, feature subsets: {len(subsets)}")
    print(f"Cartesian: {len(configs) * len(subsets)} → sampling {max_experiments}")

    # Random sample of (config, subset) pairs
    rng = np.random.default_rng(42)
    all_pairs = [(c[0], s, c[1]) for c in configs for s in subsets]
    sampled = rng.choice(len(all_pairs), size=min(max_experiments, len(all_pairs)), replace=False)

    results = []
    best_score = float("inf")
    best = None
    for i, idx in enumerate(sampled):
        config, subset, prior = all_pairs[idx]
        try:
            res = run_experiment(config, subset, team_games, prior_precision=prior)
            results.append(res)
            if res["combined_score"] < best_score:
                best_score = res["combined_score"]
                best = res
                print(f"  [{i+1}/{len(sampled)}] NEW BEST: margin={res['margin_rmse']:.2f} "
                      f"total={res['total_rmse']:.2f} winAcc={res['win_accuracy']:.1%} "
                      f"feats={res['n_features']} prior={prior}")
            elif (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(sampled)}] best so far: {best_score:.2f}")
        except Exception as e:
            print(f"  ! experiment failed: {e}")

    df = pd.DataFrame(results)
    df.to_parquet(LOG_DIR / "experiments.parquet", index=False)
    print(f"\nTotal experiments: {len(df)}")
    print(f"Best margin RMSE: {df['margin_rmse'].min():.2f}")
    print(f"Best total  RMSE: {df['total_rmse'].min():.2f}")

    # Save best config
    best_idx = df["combined_score"].idxmin()
    best_row = df.iloc[best_idx].to_dict()
    print(f"\nBEST CONFIG:")
    for k, v in best_row.items():
        print(f"  {k}: {v}")

    with open(MODEL_DIR / "best_config.json", "w") as f:
        json.dump(best_row, f, indent=2, default=str)
    print(f"\nSaved best config to {MODEL_DIR}/best_config.json")

    # Show top 5
    print("\nTop 5 configs:")
    top5 = df.nsmallest(5, "combined_score")
    print(top5[["margin_rmse", "total_rmse", "recency_decay", "playoff_weight",
                "use_opp_adjustment", "n_features", "feature_cols"]].to_string())


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n)
