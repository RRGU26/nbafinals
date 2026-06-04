# Deployment Guide

Three deployment options, ordered easiest → most control:

## Option 1: Streamlit Community Cloud (free, public — recommended)

The repo is already at https://github.com/RRGU26/nbafinals.

1. Go to **https://share.streamlit.io** and sign in with GitHub
2. Click "New app"
3. Configure:
   - **Repository:** `RRGU26/nbafinals`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Python version:** 3.11
4. (Optional) For live betting odds, click **Advanced settings → Secrets** and add:
   ```toml
   ODDS_API_KEY = "your-odds-api-key-here"
   ```
   The dashboard works without this — only the Betting page is affected.
5. Click **Deploy**. You'll get a URL like `https://nbafinals.streamlit.app`

Streamlit Cloud will:
- Install dependencies from `requirements.txt` (already in the repo)
- Auto-restart on `git push` to `main`
- Provide free TLS and a custom subdomain

## Option 2: Run locally (no deployment)

```bash
cd /Users/rr/nba-finals-2026
uv run streamlit run app.py
```

Opens at http://localhost:8501. Stop with `Ctrl+C`.

To expose on your local network (so phones can see it):
```bash
uv run streamlit run app.py --server.address 0.0.0.0
```
Then anyone on your wifi can hit `http://<your-mac-ip>:8501`.

For live odds locally:
```bash
ODDS_API_KEY=your-key uv run streamlit run app.py
```

## Option 3: Docker (for any cloud)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

Build & run:
```bash
docker build -t nbafinals .
docker run -p 8501:8501 -e ODDS_API_KEY=your-key nbafinals
```

Deploy to anywhere that runs Docker (Fly.io, Railway, Render, GCP Cloud Run, AWS ECS).

## Automated refresh — local cron (PRIMARY)

`stats.nba.com` rate-limits requests from cloud providers, so the GitHub
Action that fetches game data fails ~daily. The reliable path is a local
launchd job on your Mac (residential IPs aren't throttled).

The launchd job is already installed at
`~/Library/LaunchAgents/com.rr.nbafinals.refresh.plist`. It runs daily at
**8:00 AM local time** and executes `scripts/local_refresh.sh`:

```bash
# Check it's loaded
launchctl list | grep nbafinals

# Reinstall if you ever break it
launchctl unload ~/Library/LaunchAgents/com.rr.nbafinals.refresh.plist
launchctl load ~/Library/LaunchAgents/com.rr.nbafinals.refresh.plist

# Logs
tail -f /tmp/nbafinals_refresh.log

# Run it manually any time
./scripts/local_refresh.sh
```

The script: pulls remote → fetches game data via `nba_api` → scores any
pending prediction snapshots → re-simulates remaining games → closes
open bets → commits + pushes if anything changed.

## Tracking prediction accuracy

The Track Record page shows predicted vs actual margins/totals/win-calls
as the series progresses.

**Before each game (snapshots the current prediction):**
```bash
uv run python scorecard.py snapshot --game 1
```

**After each game completes (scores against actual result):**
```bash
uv run python fetch_data.py             # pull latest box score
uv run python scorecard.py score        # score all snapshots
```

The scorecard accumulates across all 7 games and shows:
- Win prediction accuracy
- Margin RMSE
- Total RMSE
- Which model (Analytic vs Bayesian) is performing better

## Updating data after each Finals game

The dashboard reads from `data/team_games.parquet` and `logs/series_simulation.json`.
After each game completes:

```bash
cd /Users/rr/nba-finals-2026
uv run python fetch_data.py        # pulls new boxscore (~30 sec)
uv run python series_sim.py        # re-simulates remaining games (~10 sec)
git add -A && git commit -m "Update after Game N" && git push
```

Streamlit Cloud will auto-redeploy on push.

For deeper updates (re-train the model):
```bash
uv run python autoresearch.py 500  # ~5 min — re-runs the search
uv run python series_sim.py
```

## Auto-refresh setup

Add to `~/.openclaw/cron/jobs.json` (or system cron):

```json
{
  "name": "nba-finals-refresh",
  "cron": "0 6 * * *",
  "command": "cd /Users/rr/nba-finals-2026 && uv run python fetch_data.py && uv run python series_sim.py && git add -A && git commit -m 'Daily refresh' && git push"
}
```

Runs every morning at 6am, so the dashboard reflects yesterday's game.

## Troubleshooting

- **"No live odds"** on betting page: Add your `ODDS_API_KEY` to secrets/env
- **Streamlit Cloud build fails**: Check Python version is 3.11, requirements.txt is current
- **Data is stale**: Run `fetch_data.py` and push the updated parquet files
- **Model predictions look wrong**: Re-run `autoresearch.py 400` to refresh best_config.json
