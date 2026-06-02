"""Closing Line Value (CLV) tracker.

CLV is the gold-standard metric for whether a bettor has edge: did the line
move in your direction after you bet? Consistently positive CLV → real edge,
even on losing weeks. Negative CLV → no edge, even if you're up money.

Workflow:
  1. Log a bet at the time you place it:
       uv run python clv.py log --side "NYK +4.5" --price -102 --book DraftKings
     This records what line was available across the market at that moment.
  2. Closing snapshot (before tipoff):
       uv run python clv.py close
     Records the consensus closing line for the matchup.
  3. View CLV:
       uv run python clv.py report

CLV calc for a point-spread bet:
  CLV (pts) = closing_line - recommended_line
    where positive value means you beat the close.

For SAS -4.5 at -110, if closing is SAS -6.0: CLV = -6.0 - (-4.5) = -1.5 pts of value GAINED
For NYK +4.5 at -110, if closing is NYK +3.0: CLV = +4.5 - 3.0 = +1.5 pts of value GAINED
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from odds import fetch_nba_odds, find_game, best_price

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
BET_LOG = LOG_DIR / "bet_log.jsonl"
CLV_REPORT = LOG_DIR / "clv_report.json"


def _market_snapshot(side_filter, market_key: str) -> dict | None:
    """Get consensus point and best price across books for a market side."""
    odds = fetch_nba_odds()
    if not odds:
        return None
    game = find_game(odds, "Spurs", "Knicks") or find_game(odds, "Knicks", "Spurs")
    if not game:
        return None

    best = best_price(game, market_key, side_filter)
    if not best:
        return None

    points = []
    for bk in game.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] != market_key:
                continue
            for o in m.get("outcomes", []):
                if side_filter(o):
                    if "point" in o:
                        points.append(o["point"])

    consensus_point = (sum(points) / len(points)) if points else None
    return {
        "best_price": best["price"],
        "best_book": best["book"],
        "consensus_point": consensus_point,
        "n_books": len(points) if points else len(game.get("bookmakers", [])),
        "game_commence": game.get("commence_time"),
    }


def _parse_side(side_str: str) -> dict:
    """Parse 'NYK +4.5' or 'SAS ML' or 'Over 218.5' into structured info."""
    parts = side_str.strip().split()
    if "Over" in side_str or "Under" in side_str:
        return {"market": "totals", "side": parts[0], "point": float(parts[1])}
    if parts[0] in ("NYK", "Knicks"):
        team = "New York Knicks"
    elif parts[0] in ("SAS", "Spurs"):
        team = "San Antonio Spurs"
    else:
        team = parts[0]
    if len(parts) == 1 or parts[1].upper() == "ML":
        return {"market": "h2h", "team": team}
    return {"market": "spreads", "team": team, "point": float(parts[1])}


def log_bet(side_str: str, price: int, book: str, stake: float = 0.0, notes: str = ""):
    """Record a bet at the time it's placed + capture the current market snapshot."""
    parsed = _parse_side(side_str)

    # Find current market state
    if parsed["market"] == "totals":
        snap = _market_snapshot(lambda o: o["name"] == parsed["side"], "totals")
    elif parsed["market"] == "spreads":
        snap = _market_snapshot(lambda o: o["name"] == parsed["team"], "spreads")
    else:
        snap = _market_snapshot(lambda o: o["name"] == parsed["team"], "h2h")

    row = {
        "logged_at": datetime.now().isoformat(),
        "side": side_str,
        "parsed": parsed,
        "your_price": price,
        "your_book": book,
        "stake": stake,
        "notes": notes,
        "market_at_log": snap,
        "closing_line": None,  # filled in by `close` command
        "result": None,         # filled in after the game
    }

    with open(BET_LOG, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")

    print(f"Logged: {side_str} @ {price:+d} ({book})")
    if snap:
        if "point" in parsed and snap["consensus_point"] is not None:
            print(f"  Market consensus at log time: {snap['consensus_point']:.2f}")
            print(f"  Your point vs consensus: {parsed['point'] - snap['consensus_point']:+.2f}")
        print(f"  Best across books: {snap['best_price']:+d} at {snap['best_book']}")


def close():
    """Snapshot the closing line for all open bets in the log."""
    if not BET_LOG.exists():
        print("No bets logged yet.")
        return

    rows = []
    with open(BET_LOG) as f:
        for line in f:
            rows.append(json.loads(line))

    open_bets = [r for r in rows if r.get("closing_line") is None]
    if not open_bets:
        print("All bets already have closing lines.")
        return

    updated = []
    for r in rows:
        if r.get("closing_line") is not None:
            updated.append(r)
            continue

        parsed = r["parsed"]
        if parsed["market"] == "totals":
            snap = _market_snapshot(lambda o: o["name"] == parsed["side"], "totals")
        elif parsed["market"] == "spreads":
            snap = _market_snapshot(lambda o: o["name"] == parsed["team"], "spreads")
        else:
            snap = _market_snapshot(lambda o: o["name"] == parsed["team"], "h2h")

        r["closing_line"] = snap
        r["closing_captured_at"] = datetime.now().isoformat()
        print(f"  Closed: {r['side']} → consensus {snap['consensus_point'] if snap else 'n/a'}")
        updated.append(r)

    # Rewrite
    with open(BET_LOG, "w") as f:
        for r in updated:
            f.write(json.dumps(r, default=str) + "\n")


def compute_clv(bet: dict) -> dict | None:
    """Compute CLV for a single closed bet."""
    if not bet.get("closing_line") or not bet.get("market_at_log"):
        return None
    parsed = bet["parsed"]
    if parsed["market"] not in ("spreads", "totals"):
        # ML CLV is in implied prob terms; skip for simplicity
        return None

    your_point = parsed["point"]
    closing_point = bet["closing_line"].get("consensus_point")
    if closing_point is None:
        return None

    if parsed["market"] == "totals" and parsed["side"] == "Over":
        # Over: lower closing total = better (you bet over at a lower number)
        clv_pts = your_point - closing_point
    elif parsed["market"] == "totals" and parsed["side"] == "Under":
        clv_pts = closing_point - your_point
    else:
        # Spread: depends on side. If betting team X +5 and closes at +3, +2 CLV (you got more points)
        # If betting team X -5 and closes at -7, +2 CLV (line moved your way)
        # Standard formula: CLV = your_point - closing_point IF you're getting points (positive spread)
        #                     OR: closing - your_point if giving points (negative spread)
        # Simplification: for spread, CLV = your_point - closing_point when sign matches direction.
        # We track absolute movement and direction manually:
        clv_pts = your_point - closing_point  # positive = you beat the close on a + spread

    return {
        "clv_pts": round(clv_pts, 2),
        "your_point": your_point,
        "closing_point": closing_point,
    }


def report():
    if not BET_LOG.exists():
        print("No bets logged yet.")
        return

    rows = []
    with open(BET_LOG) as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Bet log: {len(rows)} bets")
    print()

    closed = []
    for r in rows:
        clv = compute_clv(r)
        if clv:
            r["clv"] = clv
            closed.append(r)
        print(f"  {r['logged_at'][:16]}  {r['side']:<18} @ {r['your_price']:+d} ({r['your_book']})  "
              f"{'CLV ' + format(clv['clv_pts'], '+.2f') if clv else 'pending'}")

    if closed:
        avg_clv = sum(r["clv"]["clv_pts"] for r in closed) / len(closed)
        positive = sum(1 for r in closed if r["clv"]["clv_pts"] > 0)
        print(f"\nAggregate CLV ({len(closed)} closed bets):")
        print(f"  Average CLV: {avg_clv:+.2f} pts")
        print(f"  Positive CLV rate: {positive}/{len(closed)} = {positive/len(closed):.0%}")
        if avg_clv > 0.5:
            print(f"  ✅ Beating the close — model has edge signal")
        elif avg_clv > 0:
            print(f"  ⚠ Slightly positive — small sample, watch")
        else:
            print(f"  ❌ Not beating the close — model not producing real edge")

        # Save aggregate
        with open(CLV_REPORT, "w") as f:
            json.dump({
                "updated_at": datetime.now().isoformat(),
                "n_bets": len(rows),
                "n_closed": len(closed),
                "avg_clv_pts": avg_clv,
                "positive_clv_rate": positive / len(closed),
                "bets": rows,
            }, f, indent=2, default=str)
        print(f"\nSaved report to {CLV_REPORT}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    log_p = sub.add_parser("log", help="Log a bet")
    log_p.add_argument("--side", required=True, help='e.g. "NYK +4.5" or "Over 218.5" or "SAS ML"')
    log_p.add_argument("--price", type=int, required=True, help="American odds you got")
    log_p.add_argument("--book", required=True, help="Sportsbook name")
    log_p.add_argument("--stake", type=float, default=0.0)
    log_p.add_argument("--notes", default="")
    sub.add_parser("close", help="Snapshot closing line for open bets")
    sub.add_parser("report", help="Show CLV summary")
    args = p.parse_args()

    if args.cmd == "log":
        log_bet(args.side, args.price, args.book, args.stake, args.notes)
    elif args.cmd == "close":
        close()
    elif args.cmd == "report":
        report()


if __name__ == "__main__":
    main()
