"""Track Vegas line movement on the Finals matchup.

Run on a cron (every 30-60 minutes) to build a time series of spreads/totals.
Sharp money moves lines — when the spread moves 1+ pt in the hours before tipoff,
that's signal independent of our model.

Run:
    uv run python track_lines.py snapshot   # append current best lines to history
    uv run python track_lines.py summary    # print open → current movement

History file: logs/line_history.jsonl  (append-only)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from odds import fetch_nba_odds, find_game, best_price

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
HISTORY_PATH = LOG_DIR / "line_history.jsonl"


def _consensus_spread(game: dict, side_filter) -> dict | None:
    """Best price (highest payout) plus consensus point across books."""
    best = best_price(game, "spreads", side_filter)
    if not best:
        return None
    # Average the spread point across all books for a consensus line
    points = []
    for bk in game.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] != "spreads": continue
            for o in m.get("outcomes", []):
                if side_filter(o):
                    points.append(o.get("point"))
    consensus = sum(p for p in points if p is not None) / max(len([p for p in points if p is not None]), 1)
    return {**best, "consensus_point": consensus, "n_books": len(points)}


def _consensus_total(game: dict, side: str) -> dict | None:
    best = best_price(game, "totals", lambda o: o["name"] == side)
    if not best:
        return None
    points = []
    for bk in game.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] != "totals": continue
            for o in m.get("outcomes", []):
                if o["name"] == side:
                    points.append(o.get("point"))
    consensus = sum(p for p in points if p is not None) / max(len([p for p in points if p is not None]), 1)
    return {**best, "consensus_point": consensus, "n_books": len(points)}


def snapshot():
    """Pull current odds, append one row per game to history."""
    odds = fetch_nba_odds()
    if not odds:
        print("No odds available (ODDS_API_KEY missing or quota hit)")
        return None

    game = find_game(odds, "Spurs", "Knicks") or find_game(odds, "Knicks", "Spurs")
    if not game:
        print("Knicks-Spurs not in current NBA odds slate")
        return None

    home_full = game["home_team"]
    away_full = game["away_team"]

    sp_home = _consensus_spread(game, lambda o: o["name"] == home_full)
    sp_away = _consensus_spread(game, lambda o: o["name"] == away_full)
    over = _consensus_total(game, "Over")
    under = _consensus_total(game, "Under")
    ml_home = best_price(game, "h2h", lambda o: o["name"] == home_full)
    ml_away = best_price(game, "h2h", lambda o: o["name"] == away_full)

    row = {
        "captured_at": datetime.now().isoformat(),
        "commence_time": game.get("commence_time"),
        "home_team": home_full,
        "away_team": away_full,
        "home_spread_point": sp_home["consensus_point"] if sp_home else None,
        "home_spread_price": sp_home["price"] if sp_home else None,
        "away_spread_point": sp_away["consensus_point"] if sp_away else None,
        "away_spread_price": sp_away["price"] if sp_away else None,
        "total_point": over["consensus_point"] if over else None,
        "over_price": over["price"] if over else None,
        "under_price": under["price"] if under else None,
        "ml_home_price": ml_home["price"] if ml_home else None,
        "ml_away_price": ml_away["price"] if ml_away else None,
        "n_books": len(game.get("bookmakers", [])),
    }

    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"[{row['captured_at'][:19]}] {away_full} @ {home_full}")
    print(f"  Spread: {home_full[:10]} {row['home_spread_point']:+.1f}   "
          f"Total: {row['total_point']:.1f}   "
          f"ML: {row['ml_home_price']:+d} / {row['ml_away_price']:+d}   "
          f"({row['n_books']} books)")
    return row


def summary():
    """Show open → current movement."""
    if not HISTORY_PATH.exists():
        print("No history yet. Run `track_lines.py snapshot` first.")
        return

    rows = []
    with open(HISTORY_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        print("History file is empty.")
        return

    first = rows[0]
    last = rows[-1]
    home = first["home_team"]
    away = first["away_team"]

    print(f"Line history for {away} @ {home}")
    print(f"  Snapshots: {len(rows)}")
    print(f"  Open:    {first['captured_at'][:19]}")
    print(f"  Current: {last['captured_at'][:19]}")
    print()

    def fmt_move(open_val, cur_val, fmt="+.1f"):
        if open_val is None or cur_val is None:
            return "n/a"
        diff = cur_val - open_val
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        return f"{format(open_val, fmt)} → {format(cur_val, fmt)} {arrow} ({diff:+.2f})"

    print(f"  Home spread ({home[:15]}): {fmt_move(first['home_spread_point'], last['home_spread_point'])}")
    print(f"  Total points:               {fmt_move(first['total_point'], last['total_point'])}")
    print(f"  ML {home[:15]}:        {fmt_move(first['ml_home_price'], last['ml_home_price'], '+d')}")
    print(f"  ML {away[:15]}:        {fmt_move(first['ml_away_price'], last['ml_away_price'], '+d')}")

    # Sharp move detection
    spread_move = (last['home_spread_point'] or 0) - (first['home_spread_point'] or 0)
    total_move = (last['total_point'] or 0) - (first['total_point'] or 0)

    print()
    if abs(spread_move) >= 1.0:
        direction = home if spread_move < 0 else away  # if home went more negative, sharps on home
        print(f"  📈 SHARP MOVE detected on spread: {abs(spread_move):.1f} pts toward {direction}")
    elif abs(spread_move) >= 0.5:
        direction = home if spread_move < 0 else away
        print(f"  ↗ Mild spread move: {abs(spread_move):.1f} pts toward {direction}")
    else:
        print(f"  Spread stable ({spread_move:+.2f})")

    if abs(total_move) >= 1.5:
        direction = "OVER" if total_move > 0 else "UNDER"
        print(f"  📈 SHARP MOVE on total: {abs(total_move):.1f} pts toward {direction}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot", help="Append current lines to history")
    sub.add_parser("summary", help="Show open → current movement")
    args = p.parse_args()

    if args.cmd == "snapshot":
        snapshot()
    elif args.cmd == "summary":
        summary()


if __name__ == "__main__":
    main()
