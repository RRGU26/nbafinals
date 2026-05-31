"""Live NBA odds fetcher — self-contained, no external repo dependencies.

Uses The Odds API. Set ODDS_API_KEY in env or st.secrets for production.
"""

import os
import time
from typing import Optional

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def _get_api_key() -> str:
    """Try env var, then Streamlit secrets, then fall back to None."""
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key
    # Streamlit secrets (only available when running under streamlit)
    try:
        import streamlit as st
        if "ODDS_API_KEY" in st.secrets:
            return st.secrets["ODDS_API_KEY"]
    except (ImportError, Exception):
        pass
    return ""


def fetch_nba_odds() -> list[dict]:
    """Fetch current NBA odds from The Odds API. Returns empty list on failure."""
    api_key = _get_api_key()
    if not api_key:
        return []

    url = f"{ODDS_API_BASE}/sports/basketball_nba/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"Odds API failed after 3 attempts: {e}")
                return []
    return []


def find_game(odds: list[dict], home_keyword: str, away_keyword: str) -> Optional[dict]:
    """Find a specific matchup in the odds payload."""
    for g in odds:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        if home_keyword.lower() in home.lower() and away_keyword.lower() in away.lower():
            return g
        if home_keyword.lower() in away.lower() and away_keyword.lower() in home.lower():
            return g
    return None


def best_price(game: dict, market_key: str, side_filter) -> Optional[dict]:
    """Find best (highest payout) odds across all books for a side of a market."""
    best = None
    for bk in game.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m["key"] != market_key:
                continue
            for o in m.get("outcomes", []):
                if not side_filter(o):
                    continue
                if best is None or o["price"] > best["price"]:
                    best = {**o, "book": bk.get("title", bk.get("key", "?"))}
    return best


def american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def american_to_implied_prob(odds: int) -> float:
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)


if __name__ == "__main__":
    odds = fetch_nba_odds()
    print(f"Got {len(odds)} games")
    for g in odds:
        if "Knicks" in g.get("home_team", "") + g.get("away_team", "") and \
           "Spurs" in g.get("home_team", "") + g.get("away_team", ""):
            print(f"  {g['away_team']} @ {g['home_team']}")
            print(f"  Books: {len(g.get('bookmakers', []))}")
