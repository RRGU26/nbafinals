"""Combine model predictions with live odds → bet recommendation.

Pulls predictions from logs/prediction_g*.json and Finals odds via The Odds API.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

from odds import fetch_nba_odds as fetch_odds, american_to_decimal, american_to_implied_prob

LOG_DIR = Path(__file__).parent / "logs"
MARGIN_STD = 13.5
TOTAL_STD = 18.0


def best_price(game, market_key, side_filter):
    best = None
    for bk in game["bookmakers"]:
        for m in bk["markets"]:
            if m["key"] != market_key:
                continue
            for o in m["outcomes"]:
                if not side_filter(o):
                    continue
                if best is None or o["price"] > best["price"]:
                    best = {**o, "book": bk["title"]}
    return best


def kelly_fraction(p_win: float, decimal_odds: float, cap: float = 0.05) -> float:
    """Quarter-Kelly for safety."""
    b = decimal_odds - 1
    f = (p_win * b - (1 - p_win)) / b
    return max(0.0, min(f * 0.25, cap))


def main(prediction_file: str):
    with open(prediction_file) as f:
        pred = json.load(f)

    home, away = pred["home_team"], pred["away_team"]
    # Map our short codes to full names
    full = {"NYK": "New York Knicks", "SAS": "San Antonio Spurs"}
    home_full, away_full = full[home], full[away]

    print(f"Loaded prediction: {away} @ {home}\n")

    # Predictions
    analytic_margin = pred["analytic"]["margin"]
    analytic_total = pred["analytic"]["total"]
    bayes_margin = pred["bayesian"]["margin_mean"]
    bayes_total = pred["bayesian"]["total_mean"]
    consensus_margin = pred["consensus"]["margin"]
    consensus_total = pred["consensus"]["total"]

    odds = fetch_odds()
    game = next((g for g in odds if home_full in g["home_team"] and away_full in g["away_team"]), None)
    if not game:
        print(f"No live odds found for {away_full} @ {home_full}")
        return

    ml_home = best_price(game, "h2h", lambda o: o["name"] == home_full)
    ml_away = best_price(game, "h2h", lambda o: o["name"] == away_full)
    sp_home = best_price(game, "spreads", lambda o: o["name"] == home_full)
    sp_away = best_price(game, "spreads", lambda o: o["name"] == away_full)
    over = best_price(game, "totals", lambda o: o["name"] == "Over")
    under = best_price(game, "totals", lambda o: o["name"] == "Under")

    print("=" * 78)
    print(f"VEGAS LINES (best across books)")
    print("=" * 78)
    print(f"  ML  {home:>5}: {ml_home['price']:+5d} ({ml_home['book']})  → implied {american_to_implied_prob(ml_home['price']):.1%}")
    print(f"  ML  {away:>5}: {ml_away['price']:+5d} ({ml_away['book']})  → implied {american_to_implied_prob(ml_away['price']):.1%}")
    print(f"  SP  {home:>5}: {sp_home['point']:+.1f} @ {sp_home['price']:+5d} ({sp_home['book']})")
    print(f"  SP  {away:>5}: {sp_away['point']:+.1f} @ {sp_away['price']:+5d} ({sp_away['book']})")
    print(f"  Over  : {over['point']} @ {over['price']:+5d} ({over['book']})")
    print(f"  Under : {under['point']} @ {under['price']:+5d} ({under['book']})")

    print(f"\nMODEL PREDICTIONS")
    print(f"  Analytic:  margin {analytic_margin:+.1f}  total {analytic_total:.1f}  →  {home} winP {pred['analytic']['win_prob']:.1%}")
    print(f"  Bayesian:  margin {bayes_margin:+.1f}  total {bayes_total:.1f}  →  {home} winP {pred['bayesian']['win_prob']:.1%}")
    print(f"  Consensus: margin {consensus_margin:+.1f}  total {consensus_total:.1f}  →  {home} winP {pred['consensus']['win_prob']:.1%}")

    # Compute edges using CONSENSUS predictions
    print(f"\n" + "=" * 78)
    print(f"BET ANALYSIS (using CONSENSUS)")
    print("=" * 78)

    # ML edge
    p_home_consensus = pred["consensus"]["win_prob"]
    edge_ml_home = p_home_consensus - american_to_implied_prob(ml_home["price"])
    edge_ml_away = (1 - p_home_consensus) - american_to_implied_prob(ml_away["price"])

    # Spread cover prob using consensus margin
    # Home covers spread `sp_home['point']` means actual_margin > -sp_home['point']
    # e.g. SAS -5.5 means SAS margin > 5.5
    home_spread_line = sp_home["point"]
    p_home_cover = float(norm.cdf((consensus_margin - (-home_spread_line)) / MARGIN_STD))
    p_away_cover = 1 - p_home_cover
    edge_sp_home = p_home_cover - american_to_implied_prob(sp_home["price"])
    edge_sp_away = p_away_cover - american_to_implied_prob(sp_away["price"])

    # Over/Under
    over_line = over["point"]
    p_over = float(norm.cdf((consensus_total - over_line) / TOTAL_STD))
    p_under = 1 - p_over
    edge_over = p_over - american_to_implied_prob(over["price"])
    edge_under = p_under - american_to_implied_prob(under["price"])

    print(f"\n{'Bet':<22} {'Model P':<10} {'Implied':<10} {'Edge':<10} {'Kelly%':<8}")
    print("-" * 78)
    for label, p, market in [
        (f"ML {home}", p_home_consensus, ml_home),
        (f"ML {away}", 1 - p_home_consensus, ml_away),
        (f"{home} {home_spread_line:+.1f}", p_home_cover, sp_home),
        (f"{away} {sp_away['point']:+.1f}", p_away_cover, sp_away),
        (f"Over {over_line}", p_over, over),
        (f"Under {under['point']}", p_under, under),
    ]:
        dec = american_to_decimal(market["price"])
        imp = american_to_implied_prob(market["price"])
        edge = p - imp
        kelly = kelly_fraction(p, dec) * 100
        flag = " ✓" if edge > 0.03 else ""
        print(f"{label:<22} {p:<10.1%} {imp:<10.1%} {edge:+.1%}{'':<3} {kelly:.1f}%{flag}")

    # Honest assessment
    print(f"\n" + "=" * 78)
    print(f"RECOMMENDATION")
    print("=" * 78)
    print(f"""
Analytic model says NYK by {-analytic_margin:.1f}. Bayesian says {home} by {bayes_margin:.1f}.
That's a {abs(analytic_margin - bayes_margin):.0f}-point disagreement between two models built on
the same data. With only 16 playoff games per team, NEITHER model is well-calibrated.

Vegas has {home} -{abs(sp_home['point']):.1f}, implying they see {home} as clearly better.
Vegas saw 80+ regular season games AND the actual playoff matchups. They have
more info than our model on injuries, lineups, and matchup specifics.

When v1 model disagrees with Vegas by 6+ points, the safe assumption is Vegas
is right. Bet small or skip for Game 1; update model after we see live data.

SUGGESTED ACTION FOR GAME 1:
  - Either SKIP entirely (let model see Game 1 result first), or
  - Bet small ($25-50) on {away} +{sp_away['point']:.1f} if you trust the analytic model
    (model says {away} wins, so +{sp_away['point']:.1f} is a free buffer)
  - Watch Game 1, then re-evaluate Game 2 with one extra data point
""")


if __name__ == "__main__":
    pred_file = sys.argv[1] if len(sys.argv) > 1 else "logs/prediction_g1_SAS_vs_NYK.json"
    main(pred_file)
