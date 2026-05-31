"""Template-based commentary generator.

Turns model outputs into analyst-style prose. No LLM dependency.
"""

from __future__ import annotations
import random


def _comparator(a: float, b: float, threshold: float = 1.0) -> str:
    diff = a - b
    if abs(diff) < threshold:
        return "essentially even"
    if diff > 8:
        return "decisively better"
    if diff > 4:
        return "clearly better"
    if diff > 0:
        return "marginally better"
    if diff > -4:
        return "marginally worse"
    if diff > -8:
        return "clearly worse"
    return "decisively worse"


def series_overview(sim_result: dict) -> str:
    """One-paragraph series overview from simulation."""
    p_nyk = sim_result["simulation"]["p_nyk_wins"]
    expected_games = sim_result["simulation"]["expected_games"]
    most_likely = max(sim_result["simulation"]["outcomes"].items(), key=lambda x: x[1])

    confidence = (
        "strongly" if abs(p_nyk - 0.5) > 0.20 else
        "moderately" if abs(p_nyk - 0.5) > 0.10 else
        "slightly"
    )
    favorite = "Knicks" if p_nyk > 0.5 else "Spurs"
    fav_prob = p_nyk if p_nyk > 0.5 else (1 - p_nyk)

    return (
        f"Our model {confidence} favors the **{favorite}** to win the series "
        f"({fav_prob:.0%}), with the most likely outcome being **{most_likely[0]}** "
        f"at {most_likely[1]:.0%}. Expected series length: **{expected_games:.1f} games**. "
    )


def matchup_analysis(home_form: dict, away_form: dict, home: str, away: str) -> str:
    """Compare two teams' recent form."""
    h_net = home_form["net_rating"]
    a_net = away_form["net_rating"]
    h_pace = home_form["pace"]
    a_pace = away_form["pace"]

    net_diff = h_net - a_net

    parts = []
    # Strength comparison
    if abs(net_diff) > 8:
        better, worse = (home, away) if net_diff > 0 else (away, home)
        margin = abs(net_diff)
        parts.append(
            f"The {better} have been **{margin:.0f} points per 100 possessions better** "
            f"than the {worse} over the recency-weighted window the model uses."
        )
    elif abs(net_diff) > 3:
        better = home if net_diff > 0 else away
        parts.append(
            f"The {better} hold the edge in net rating, though it's not insurmountable."
        )
    else:
        parts.append("Net ratings are nearly identical — this projects as a closely matched series.")

    # Offense
    h_ortg, a_ortg = home_form["off_rating"], away_form["off_rating"]
    if abs(h_ortg - a_ortg) > 5:
        better = home if h_ortg > a_ortg else away
        parts.append(f"Offensively, the {better} have been more efficient "
                     f"({max(h_ortg,a_ortg):.0f} vs {min(h_ortg,a_ortg):.0f} ORtg).")

    # Defense
    h_drtg, a_drtg = home_form["def_rating"], away_form["def_rating"]
    if abs(h_drtg - a_drtg) > 5:
        better = home if h_drtg < a_drtg else away
        parts.append(f"Defensively, the {better} have been tighter "
                     f"({min(h_drtg,a_drtg):.0f} vs {max(h_drtg,a_drtg):.0f} DRtg).")

    # Pace
    if abs(h_pace - a_pace) > 3:
        faster = home if h_pace > a_pace else away
        parts.append(
            f"There's a meaningful pace gap — the {faster} play faster "
            f"({max(h_pace,a_pace):.1f} vs {min(h_pace,a_pace):.1f} possessions/48). "
            f"Whoever dictates tempo gains an edge."
        )

    # Momentum
    h_mom = home_form["momentum_l5"]
    a_mom = away_form["momentum_l5"]
    if h_mom > a_mom + 10:
        parts.append(f"The {home} enter the series scorching hot, winning by an average of "
                     f"**{h_mom:+.1f} points** over their last five games.")
    elif a_mom > h_mom + 10:
        parts.append(f"The {away} enter the series scorching hot, winning by an average of "
                     f"**{a_mom:+.1f} points** over their last five games.")

    return " ".join(parts)


def game_commentary(per_game: dict, vegas_line: dict | None = None) -> str:
    """One-paragraph commentary on a single game."""
    game_num = per_game["game"]
    home = per_game["home"]
    away = per_game["away"]
    nyk_p = per_game["nyk_win_prob"]
    margin = per_game["predicted_margin_home"]
    total = per_game["predicted_total"]
    analytic_m = per_game["analytic_margin_home"]
    bayes_m = per_game["bayes_margin_home"]

    parts = []
    favorite = "Knicks" if nyk_p > 0.5 else "Spurs"
    fav_prob = nyk_p if nyk_p > 0.5 else (1 - nyk_p)
    parts.append(
        f"**Game {game_num} ({away} @ {home}):** Model gives the **{favorite} a {fav_prob:.0%}** chance, "
        f"projecting a final margin of {margin:+.1f} from the home perspective with a total around {total:.0f}."
    )

    # Note model agreement
    if abs(analytic_m - bayes_m) > 8:
        parts.append(
            f"There's significant disagreement between the two model components "
            f"(analytic: {analytic_m:+.1f}, Bayesian: {bayes_m:+.1f}) — "
            f"interpret the prediction with extra uncertainty."
        )

    # Compare to Vegas if available
    if vegas_line:
        v_spread = vegas_line.get("home_spread")
        v_total = vegas_line.get("total")
        if v_spread is not None:
            model_implied_spread = -margin  # if margin home is +5, model "spreads" home -5
            gap = model_implied_spread - v_spread
            if abs(gap) > 3:
                side = "home" if gap < 0 else "away"
                parts.append(
                    f"Vegas opened {home} at {v_spread:+.1f}; model implies {model_implied_spread:+.1f} — "
                    f"a {abs(gap):.1f}-point gap favoring the **{side} team**."
                )
        if v_total is not None and abs(total - v_total) > 5:
            side = "over" if total > v_total else "under"
            parts.append(f"Total disagreement: model {total:.0f} vs Vegas {v_total:.1f} — lean **{side}** if you trust the model.")

    return " ".join(parts)


def bet_recommendation_commentary(edges: list[dict]) -> str:
    """Synthesize betting recommendations from edge analysis."""
    profitable = [e for e in edges if e["edge"] > 0.03]
    if not profitable:
        return ("No bets clear our 3% edge threshold tonight. **Skip this slate** — "
                "no edge means no bet, regardless of how strong the model feels.")

    parts = ["**Recommended bets:**"]
    for e in sorted(profitable, key=lambda x: -x["edge"]):
        parts.append(
            f"- **{e['name']}** at {e['american_odds']:+d} ({e['book']}): "
            f"model {e['model_prob']:.1%} vs implied {e['implied_prob']:.1%} = "
            f"**+{e['edge']*100:.1f}% edge**, Kelly {e['kelly_pct']:.1f}%"
        )

    if len(profitable) > 1:
        parts.append(
            "\n_Consider bet sizing carefully when multiple legs share correlation "
            "(e.g., favoring NYK ML + NYK spread = stacked exposure to the same outcome)._"
        )
    return "\n".join(parts)


def methodology_blurb() -> str:
    """Static methodology explanation."""
    return """
### How the model works

This is a **two-model consensus**:

1. **Analytic** — canonical power-rating formula:
   `predicted_margin = HOME_COURT + (home_net_rating - away_net_rating)`
   Net ratings are recency-weighted (last 30 games, playoffs counted 5x).

2. **Bayesian regression** — linear model with conjugate Normal-InverseGamma prior.
   Features (selected by autoresearch over 400 experiments): `is_home`, `net_diff`, `pace_avg`.
   Strong prior (precision=100) to prevent overfitting on the small playoff sample.

### Key limitations

- **Tiny sample**: only 32 playoff games (16 per team) for the regime we care about.
  The Bayesian model's coefficients are heavily shrunk toward zero — the analytic
  formula is doing most of the real work.

- **Trained on RS data**: the regular-season component (192 rows) doesn't know
  about playoff-specific dynamics (slower pace, tighter defense, shorter rotations).

- **Vegas usually wins**: bookmakers have injury reports, lineup info, and
  matchup-specific data we don't. When our model disagrees with Vegas by 5+
  points, the default assumption should be that Vegas is right.

- **No player-level modeling**: we predict team-level outcomes only. Props
  (PRA overs, points overs) are likely a better edge but require separate data.

### Refreshing after each game

Run `uv run python fetch_data.py` after every Finals game completes — this
re-pulls the latest box score. Then re-run `predict_game.py` and `series_sim.py`
to update predictions with the new evidence.
""".strip()


def full_writeup(sim_result: dict, h_form: dict, a_form: dict, home: str, away: str) -> dict:
    """Build a complete writeup as a dict of named sections."""
    return {
        "headline": series_overview(sim_result),
        "matchup": matchup_analysis(h_form, a_form, home, away),
        "game_previews": [
            game_commentary(p) for p in sim_result["per_game"]
        ],
        "methodology": methodology_blurb(),
    }


if __name__ == "__main__":
    import json
    with open("logs/series_simulation.json") as f:
        sim = json.load(f)
    # Need team form too
    from features import FeatureConfig, load_data, league_baselines, team_features
    from predict_game import load_best_config, TEAM_IDS
    import pandas as pd

    config, _, _, _ = load_best_config()
    team_games, _ = load_data()
    baselines = league_baselines(team_games)
    as_of = pd.Timestamp("2026-06-04")
    sas_form = team_features(TEAM_IDS["SAS"], as_of, team_games, baselines, config)
    nyk_form = team_features(TEAM_IDS["NYK"], as_of, team_games, baselines, config)

    print(series_overview(sim))
    print()
    print(matchup_analysis(sas_form, nyk_form, "Spurs", "Knicks"))
    print()
    for g in sim["per_game"]:
        print(game_commentary(g))
        print()
