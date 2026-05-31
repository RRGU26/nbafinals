"""Prediction tracker — snapshot predictions and score them after games complete.

Workflow:
  1. Before each Finals game: snapshot the current model prediction
       uv run python scorecard.py snapshot --game N
  2. After game finishes: pull actual result, score the prediction
       uv run python scorecard.py score

The scorecard accumulates over the series so the dashboard can show:
  - Per-game predicted vs actual margin/total
  - Cumulative win prediction accuracy
  - Total/margin RMSE so far
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from features import load_data
from predict_game import TEAM_IDS, predict

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"
SNAPSHOTS_DIR = LOG_DIR / "prediction_snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
SCORECARD_PATH = LOG_DIR / "scorecard.json"

# 2026 Finals home court pattern (higher seed = SAS, 2-2-1-1-1)
HOME_PATTERN = {1: "SAS", 2: "SAS", 3: "NYK", 4: "NYK", 5: "SAS", 6: "NYK", 7: "SAS"}


def snapshot(game_num: int):
    """Snapshot the current prediction for a game BEFORE it plays."""
    if game_num not in HOME_PATTERN:
        print(f"Invalid game number: {game_num}")
        return

    home = HOME_PATTERN[game_num]
    away = "NYK" if home == "SAS" else "SAS"
    print(f"Snapshotting Game {game_num}: {away} @ {home}")

    # Use today as as_of (since we predict the night-of)
    pred = predict(home, away, as_of_date=pd.Timestamp.now().normalize(), game_num=game_num)

    out = {
        "game_num": game_num,
        "snapshot_at": datetime.now().isoformat(),
        "home_team": home,
        "away_team": away,
        "home_win_prob": pred["consensus"]["win_prob"],
        "predicted_margin_home": pred["consensus"]["margin"],
        "predicted_total": pred["consensus"]["total"],
        "analytic_margin": pred["analytic"]["margin"],
        "analytic_total": pred["analytic"]["total"],
        "bayesian_margin": pred["bayesian"]["margin_mean"],
        "bayesian_total": pred["bayesian"]["total_mean"],
    }

    snap_path = SNAPSHOTS_DIR / f"game_{game_num}_snapshot.json"
    with open(snap_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {snap_path}")
    return out


def get_actual_result(home_team_short: str, away_team_short: str, after_date: pd.Timestamp) -> dict | None:
    """Find the most recent NYK-SAS game in team_games.parquet after `after_date`."""
    team_games, _ = load_data()
    nyk_id = TEAM_IDS["NYK"]
    sas_id = TEAM_IDS["SAS"]

    home_id = TEAM_IDS[home_team_short]
    away_id = TEAM_IDS[away_team_short]

    # Find games where home_id was at home vs away_id
    h2h = team_games[
        (team_games["TEAM_ID"] == home_id) &
        (team_games["OPP_TEAM_ID"] == away_id) &
        (team_games["IS_HOME"]) &
        (team_games["GAME_DATE"] >= after_date)
    ].sort_values("GAME_DATE")

    if len(h2h) == 0:
        return None

    g = h2h.iloc[0]
    return {
        "game_date": str(g["GAME_DATE"].date()),
        "home_pts": int(g["PTS"]),
        "away_pts": int(g["OPP_PTS"]),
        "actual_margin": int(g["MARGIN"]),
        "actual_total": int(g["TOTAL_POINTS"]),
        "home_won": g["WL"] == "W",
        "game_id": g["GAME_ID"],
    }


def score():
    """Score all snapshotted predictions that have results available."""
    snapshots = sorted(SNAPSHOTS_DIR.glob("game_*_snapshot.json"))
    if not snapshots:
        print("No snapshots to score yet.")
        return

    results = []
    for snap_path in snapshots:
        with open(snap_path) as f:
            snap = json.load(f)
        snap_date = pd.Timestamp(snap["snapshot_at"][:10])
        actual = get_actual_result(snap["home_team"], snap["away_team"], snap_date)

        if not actual:
            print(f"  Game {snap['game_num']}: no result yet (snapshotted {snap['snapshot_at'][:10]})")
            results.append({**snap, "scored": False})
            continue

        # Compute errors
        margin_err = actual["actual_margin"] - snap["predicted_margin_home"]
        total_err = actual["actual_total"] - snap["predicted_total"]
        win_correct = (snap["home_win_prob"] > 0.5) == actual["home_won"]

        scored = {
            **snap,
            "scored": True,
            **actual,
            "margin_error": margin_err,
            "total_error": total_err,
            "win_pred_correct": win_correct,
            "analytic_margin_err": actual["actual_margin"] - snap["analytic_margin"],
            "bayesian_margin_err": actual["actual_margin"] - snap["bayesian_margin"],
        }
        results.append(scored)

        symbol = "✓" if win_correct else "✗"
        print(f"  Game {snap['game_num']}: {snap['away_team']} {actual['away_pts']} @ "
              f"{snap['home_team']} {actual['home_pts']}  "
              f"({snap['away_team'] if actual['actual_margin']<0 else snap['home_team']} by {abs(actual['actual_margin'])})")
        print(f"    Predicted: {snap['predicted_margin_home']:+.1f}, actual {actual['actual_margin']:+d} "
              f"(err {margin_err:+.1f})   Win pred {symbol}")
        print(f"    Total predicted: {snap['predicted_total']:.0f}, actual {actual['actual_total']} "
              f"(err {total_err:+.1f})")

    # Summary stats
    scored = [r for r in results if r["scored"]]
    if scored:
        win_acc = sum(1 for r in scored if r["win_pred_correct"]) / len(scored)
        margin_rmse = (sum((r["margin_error"]) ** 2 for r in scored) / len(scored)) ** 0.5
        total_rmse = (sum((r["total_error"]) ** 2 for r in scored) / len(scored)) ** 0.5
        analytic_rmse = (sum((r["analytic_margin_err"]) ** 2 for r in scored) / len(scored)) ** 0.5
        bayes_rmse = (sum((r["bayesian_margin_err"]) ** 2 for r in scored) / len(scored)) ** 0.5
    else:
        win_acc = margin_rmse = total_rmse = analytic_rmse = bayes_rmse = None

    out = {
        "updated_at": datetime.now().isoformat(),
        "n_games_scored": len(scored),
        "n_games_pending": len(results) - len(scored),
        "win_accuracy": win_acc,
        "margin_rmse": margin_rmse,
        "total_rmse": total_rmse,
        "analytic_margin_rmse": analytic_rmse,
        "bayesian_margin_rmse": bayes_rmse,
        "games": results,
    }

    with open(SCORECARD_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)

    if scored:
        print(f"\nSummary ({len(scored)} games scored):")
        print(f"  Win prediction accuracy: {win_acc:.1%}")
        print(f"  Margin RMSE: {margin_rmse:.1f}")
        print(f"  Total RMSE: {total_rmse:.1f}")
        print(f"  Analytic-only RMSE: {analytic_rmse:.1f}")
        print(f"  Bayesian-only RMSE: {bayes_rmse:.1f}")

    print(f"\nScorecard saved to {SCORECARD_PATH}")
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    snap_p = sub.add_parser("snapshot", help="Save the current model's prediction for a game before tipoff")
    snap_p.add_argument("--game", type=int, required=True)
    sub.add_parser("score", help="Score snapshots against actual results")
    args = p.parse_args()

    if args.cmd == "snapshot":
        snapshot(args.game)
    elif args.cmd == "score":
        score()


if __name__ == "__main__":
    main()
