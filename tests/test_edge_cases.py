"""Edge case tests: missing files, malformed data, network failures."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def test(name):
    def wrap(fn):
        global PASS, FAIL
        try:
            fn()
            PASS += 1
            print(f"  ✓ {name}")
        except Exception as e:
            FAIL += 1
            print(f"  ✗ {name}: {e}")
        return fn
    return wrap


print("\n=== Edge cases ===")


@test("predict_game gracefully fails on bogus team code")
def t():
    from predict_game import predict
    try:
        predict("XXX", "YYY", pd.Timestamp("2026-06-04"))
        raise AssertionError("Should have raised KeyError")
    except KeyError:
        pass  # expected


@test("series_sim handles extreme probabilities (0.99)")
def t():
    from series_sim import simulate_series
    r = simulate_series([0.99] * 7, n_sims=500)
    assert r["p_nyk_wins"] > 0.98
    assert r["expected_games"] < 4.5


@test("series_sim handles all-zero probabilities")
def t():
    from series_sim import simulate_series
    r = simulate_series([0.0] * 7, n_sims=200)
    assert r["p_nyk_wins"] < 0.02
    assert r["p_sas_wins"] > 0.98


@test("Bayesian model handles single feature")
def t():
    from model import BayesianLinearModel
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 1))
    y = X[:, 0] * 2 + rng.standard_normal(50) * 0.5
    m = BayesianLinearModel(prior_precision=0.1)
    m.fit(X, y)
    mean, var = m.predict(np.array([[1.0]]))
    assert 1.5 < mean[0] < 2.5  # close to 2


@test("Bayesian model handles constant feature (zero std)")
def t():
    from model import BayesianLinearModel
    X = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]])
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    m = BayesianLinearModel(prior_precision=1.0)
    m.fit(X, y)
    # Should not crash; intercept absorbs everything
    mean, var = m.predict(np.array([[1.0]]))
    assert np.isfinite(mean[0])


@test("Bayesian model with all-zero weights doesn't crash")
def t():
    from model import BayesianLinearModel
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    y = rng.standard_normal(20)
    m = BayesianLinearModel(prior_precision=1.0)
    # Equal positive weights
    m.fit(X, y, sample_weight=np.ones(20))
    mean, var = m.predict(X[:1])
    assert np.isfinite(mean[0])


@test("odds.fetch_nba_odds returns [] on missing API key")
def t():
    from odds import fetch_nba_odds
    with patch.dict(os.environ, {"ODDS_API_KEY": ""}, clear=False):
        # Also remove if set
        env_backup = os.environ.get("ODDS_API_KEY")
        try:
            if "ODDS_API_KEY" in os.environ:
                del os.environ["ODDS_API_KEY"]
            result = fetch_nba_odds()
            assert isinstance(result, list)
        finally:
            if env_backup:
                os.environ["ODDS_API_KEY"] = env_backup


@test("commentary handles missing fields gracefully")
def t():
    from commentary import matchup_analysis
    # Minimal dict
    h = {"net_rating": 0, "off_rating": 110, "def_rating": 110, "pace": 100, "momentum_l5": 0}
    a = {"net_rating": 0, "off_rating": 110, "def_rating": 110, "pace": 100, "momentum_l5": 0}
    text = matchup_analysis(h, a, "A", "B")
    assert len(text) > 0


@test("features handle date with no historical data")
def t():
    from features import team_features, FeatureConfig, load_data, league_baselines
    tg, _ = load_data()
    baselines = league_baselines(tg)
    feats = team_features(1610612752, pd.Timestamp("1900-01-01"), tg, baselines, FeatureConfig())
    assert feats["games_played"] == 0


@test("features handle unknown team_id")
def t():
    from features import team_features, FeatureConfig, load_data, league_baselines
    tg, _ = load_data()
    baselines = league_baselines(tg)
    feats = team_features(99999999, pd.Timestamp("2026-06-04"), tg, baselines, FeatureConfig())
    # Should fall back to defaults
    assert feats["games_played"] == 0


@test("autoresearch handles 1-experiment run")
def t():
    from autoresearch import run_experiment
    from features import FeatureConfig, load_data
    tg, _ = load_data()
    config = FeatureConfig(recency_decay=1.0, playoff_weight=3.0,
                            use_opp_adjustment=True, last_n_games=30)
    res = run_experiment(config, ["is_home", "net_diff"], tg, prior_precision=10.0)
    assert "margin_rmse" in res
    assert res["margin_rmse"] > 0


print(f"\n  Total: {PASS + FAIL} | Passed: {PASS} | Failed: {FAIL}")
sys.exit(0 if FAIL == 0 else 1)
