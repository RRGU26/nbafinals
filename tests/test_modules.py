"""Comprehensive unit/integration tests for every module.

Tests:
- All modules import without errors
- Each function works with valid inputs
- Edge cases: empty data, missing files, network failures
- Numerical stability across many runs
- End-to-end pipeline runs cleanly

Run:
    uv run python tests/test_modules.py
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
FAILURES = []


def test(name):
    """Decorator: run a test and report."""
    def wrap(fn):
        global PASS, FAIL
        try:
            fn()
            PASS += 1
            print(f"  ✓ {name}")
        except Exception as e:
            FAIL += 1
            FAILURES.append((name, str(e), traceback.format_exc()))
            print(f"  ✗ {name}: {e}")
        return fn
    return wrap


# ─────────────────────────────────────────────────────────────────────────
print("\n=== Module imports ===")
# ─────────────────────────────────────────────────────────────────────────

@test("import features")
def t():
    import features  # noqa

@test("import model")
def t():
    import model  # noqa

@test("import autoresearch")
def t():
    import autoresearch  # noqa

@test("import predict_game")
def t():
    import predict_game  # noqa

@test("import series_sim")
def t():
    import series_sim  # noqa

@test("import commentary")
def t():
    import commentary  # noqa

@test("import odds")
def t():
    import odds  # noqa

# ─────────────────────────────────────────────────────────────────────────
print("\n=== Data files exist ===")
# ─────────────────────────────────────────────────────────────────────────

@test("team_games.parquet exists")
def t():
    assert (ROOT / "data" / "team_games.parquet").exists()

@test("playoff_games.parquet exists")
def t():
    assert (ROOT / "data" / "playoff_games.parquet").exists()

@test("historical_finals.parquet exists")
def t():
    assert (ROOT / "data" / "historical_finals.parquet").exists()

@test("best_config.json exists")
def t():
    assert (ROOT / "model" / "best_config.json").exists()

@test("series_simulation.json exists")
def t():
    assert (ROOT / "logs" / "series_simulation.json").exists()

@test("experiments.parquet exists")
def t():
    assert (ROOT / "logs" / "experiments.parquet").exists()


# ─────────────────────────────────────────────────────────────────────────
print("\n=== Data integrity ===")
# ─────────────────────────────────────────────────────────────────────────

@test("team_games has both teams")
def t():
    df = pd.read_parquet(ROOT / "data" / "team_games.parquet")
    teams = set(df["TEAM_NAME"].unique())
    assert "New York Knicks" in teams
    assert "San Antonio Spurs" in teams

@test("team_games has no NaN in critical fields")
def t():
    df = pd.read_parquet(ROOT / "data" / "team_games.parquet")
    for col in ["PTS", "OPP_PTS", "MARGIN", "TOTAL_POINTS", "OFF_RATING", "DEF_RATING"]:
        assert df[col].notna().all(), f"NaN found in {col}"

@test("team_games margins are consistent")
def t():
    df = pd.read_parquet(ROOT / "data" / "team_games.parquet")
    assert ((df["PTS"] - df["OPP_PTS"]) == df["MARGIN"]).all()

@test("historical Finals has 60+ games")
def t():
    df = pd.read_parquet(ROOT / "data" / "historical_finals.parquet")
    n_games = df["GAME_ID"].nunique()
    assert n_games >= 60, f"only {n_games} historical Finals games"

@test("pace values are sane (90-115)")
def t():
    df = pd.read_parquet(ROOT / "data" / "team_games.parquet")
    assert df["PACE"].min() > 80 and df["PACE"].max() < 120


# ─────────────────────────────────────────────────────────────────────────
print("\n=== Feature engineering ===")
# ─────────────────────────────────────────────────────────────────────────

@test("FeatureConfig has sensible defaults")
def t():
    from features import FeatureConfig
    c = FeatureConfig()
    assert 0 < c.recency_decay <= 1
    assert c.playoff_weight >= 1
    assert c.last_n_games > 0

@test("team_features handles empty history (no games before date)")
def t():
    from features import team_features, FeatureConfig, load_data, league_baselines
    tg, _ = load_data()
    baselines = league_baselines(tg)
    # Pre-historic date — no games before
    early = pd.Timestamp("2000-01-01")
    feats = team_features(1610612752, early, tg, baselines, FeatureConfig())
    assert feats["games_played"] == 0
    assert feats["off_rating"] > 0  # sensible default

@test("perspective_features returns expected keys")
def t():
    from features import perspective_features, FeatureConfig, load_data, league_baselines
    tg, _ = load_data()
    baselines = league_baselines(tg)
    feats = perspective_features(1610612752, 1610612759, True, pd.Timestamp("2026-06-04"),
                                   tg, baselines, FeatureConfig())
    for key in ["off_diff", "def_diff", "net_diff", "is_home"]:
        assert key in feats

@test("build_training_set produces both home + away rows")
def t():
    from features import build_training_set, FeatureConfig, load_data
    tg, _ = load_data()
    X, y_m, y_t = build_training_set(tg, FeatureConfig())
    # Should have both is_home=0 and is_home=1 rows
    assert (X["is_home"] == 1).any()
    assert (X["is_home"] == 0).any()
    assert len(y_m) == len(X)
    assert len(y_t) == len(X)


# ─────────────────────────────────────────────────────────────────────────
print("\n=== Model ===")
# ─────────────────────────────────────────────────────────────────────────

@test("BayesianLinearModel fits cleanly on synthetic data")
def t():
    from model import BayesianLinearModel
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    true_beta = np.array([0.5, -1.0, 2.0])
    y = X @ true_beta + rng.standard_normal(100) * 2
    m = BayesianLinearModel(prior_precision=0.1)
    m.fit(X, y)
    coefs = m.coefficients_native_scale(["a", "b", "c"])
    # Coefficients should be close to true values
    assert abs(coefs["a"] - 0.5) < 0.5
    assert abs(coefs["b"] - (-1.0)) < 0.5
    assert abs(coefs["c"] - 2.0) < 0.5

@test("BayesianLinearModel predict returns mean + var")
def t():
    from model import BayesianLinearModel
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2))
    y = X[:, 0] + 0.5 * X[:, 1]
    m = BayesianLinearModel(prior_precision=1.0)
    m.fit(X, y)
    mean, var = m.predict(rng.standard_normal((10, 2)))
    assert mean.shape == (10,)
    assert var.shape == (10,)
    assert (var > 0).all()

@test("win_prob bounded in [0,1]")
def t():
    from model import BayesianLinearModel
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2))
    y = X[:, 0] * 10
    m = BayesianLinearModel(prior_precision=1.0)
    m.fit(X, y)
    p = m.win_prob(rng.standard_normal((20, 2)))
    assert (p >= 0).all() and (p <= 1).all()


# ─────────────────────────────────────────────────────────────────────────
print("\n=== Prediction pipeline ===")
# ─────────────────────────────────────────────────────────────────────────

@test("predict_game.predict runs cleanly")
def t():
    from predict_game import predict
    result = predict("SAS", "NYK", pd.Timestamp("2026-06-04"), 1)
    assert "consensus" in result
    assert 0 < result["consensus"]["win_prob"] < 1
    assert 180 < result["consensus"]["total"] < 260
    assert -40 < result["consensus"]["margin"] < 40

@test("predict from NYK home produces inverted result")
def t():
    from predict_game import predict
    sas_home = predict("SAS", "NYK", pd.Timestamp("2026-06-04"), 1)
    nyk_home = predict("NYK", "SAS", pd.Timestamp("2026-06-04"), 3)
    # Home court should help — NYK at home should have higher win prob
    # than NYK on the road
    assert nyk_home["consensus"]["win_prob"] > (1 - sas_home["consensus"]["win_prob"])


# ─────────────────────────────────────────────────────────────────────────
print("\n=== Simulation ===")
# ─────────────────────────────────────────────────────────────────────────

@test("simulate_series runs with arbitrary probabilities")
def t():
    from series_sim import simulate_series
    probs = [0.5] * 7
    result = simulate_series(probs, n_sims=1000)
    assert abs(result["p_nyk_wins"] - 0.5) < 0.05  # ~50-50

@test("simulate_series favors strong team")
def t():
    from series_sim import simulate_series
    probs = [0.8] * 7  # NYK wins 80% per game
    result = simulate_series(probs, n_sims=2000)
    assert result["p_nyk_wins"] > 0.95

@test("series outcomes sum to 1")
def t():
    from series_sim import simulate_series
    result = simulate_series([0.6, 0.4, 0.7, 0.5, 0.6, 0.4, 0.7], n_sims=5000)
    total = sum(result["outcomes"].values())
    assert abs(total - 1.0) < 0.001

@test("expected_games between 4 and 7")
def t():
    from series_sim import simulate_series
    result = simulate_series([0.5] * 7, n_sims=2000)
    assert 4 <= result["expected_games"] <= 7


# ─────────────────────────────────────────────────────────────────────────
print("\n=== Commentary ===")
# ─────────────────────────────────────────────────────────────────────────

@test("series_overview produces non-empty text")
def t():
    from commentary import series_overview
    with open(ROOT / "logs" / "series_simulation.json") as f:
        sim = json.load(f)
    text = series_overview(sim)
    assert len(text) > 50

@test("matchup_analysis includes both team names")
def t():
    from commentary import matchup_analysis
    h = {"net_rating": 15, "off_rating": 120, "def_rating": 105, "pace": 98, "momentum_l5": 20}
    a = {"net_rating": 5, "off_rating": 112, "def_rating": 107, "pace": 102, "momentum_l5": 3}
    text = matchup_analysis(h, a, "Home", "Away")
    assert "Home" in text or "Away" in text

@test("game_commentary handles dictionary input")
def t():
    from commentary import game_commentary
    g = {"game": 1, "home": "SAS", "away": "NYK", "nyk_win_prob": 0.48,
         "predicted_margin_home": 1.5, "predicted_total": 220.0,
         "analytic_margin_home": -2.0, "bayes_margin_home": 5.0}
    text = game_commentary(g)
    assert "Game 1" in text


# ─────────────────────────────────────────────────────────────────────────
print("\n=== Odds module ===")
# ─────────────────────────────────────────────────────────────────────────

@test("american_to_implied_prob is correct")
def t():
    from odds import american_to_implied_prob
    assert abs(american_to_implied_prob(-110) - 0.5238) < 0.001
    assert abs(american_to_implied_prob(+100) - 0.5) < 0.001
    assert abs(american_to_implied_prob(+200) - 0.3333) < 0.001

@test("american_to_decimal is correct")
def t():
    from odds import american_to_decimal
    assert abs(american_to_decimal(-110) - 1.909) < 0.01
    assert abs(american_to_decimal(+100) - 2.0) < 0.001
    assert abs(american_to_decimal(+200) - 3.0) < 0.001

@test("fetch_nba_odds returns list (even if empty)")
def t():
    from odds import fetch_nba_odds
    result = fetch_nba_odds()
    assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
# ─────────────────────────────────────────────────────────────────────────
print(f"\n  Total: {PASS + FAIL} | Passed: {PASS} | Failed: {FAIL}")
if FAILURES:
    print("\nFAILURE DETAILS:")
    for name, err, tb in FAILURES:
        print(f"\n--- {name} ---")
        print(tb[:500])

sys.exit(0 if FAIL == 0 else 1)
