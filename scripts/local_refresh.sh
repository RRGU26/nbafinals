#!/usr/bin/env bash
# Local post-game refresh — runs on the Mac where nba_api isn't throttled.
#
# Install with: see DEPLOY.md "Local cron setup"
#
# Logs to /tmp/nbafinals_refresh.log

set -e

REPO="/Users/rr/nba-finals-2026"
LOG="/tmp/nbafinals_refresh.log"

echo "=== $(date) ===" >> "$LOG"
cd "$REPO"

# Pull any remote changes first (the GH Actions push line snapshots etc.)
git pull --rebase --quiet >> "$LOG" 2>&1 || true

# Fetch latest games (works from residential IP)
uv run python fetch_data.py >> "$LOG" 2>&1

# Score any pending prediction snapshots against actuals
uv run python scorecard.py score >> "$LOG" 2>&1 || echo "no snapshots to score" >> "$LOG"

# Re-simulate remaining games with new evidence
uv run python series_sim.py >> "$LOG" 2>&1

# Close any open bets (snapshot closing lines for CLV)
uv run python clv.py close >> "$LOG" 2>&1 || echo "no open bets" >> "$LOG"

# Commit + push if anything changed
if ! git diff --quiet data/ logs/; then
    git add data/ logs/
    git commit -m "Local post-game refresh $(date -u +%Y-%m-%d) [auto]" >> "$LOG" 2>&1
    git push >> "$LOG" 2>&1
    echo "✓ committed and pushed" >> "$LOG"
else
    echo "no changes" >> "$LOG"
fi
echo "" >> "$LOG"
