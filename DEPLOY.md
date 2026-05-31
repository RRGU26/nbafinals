# Deployment Guide

The dashboard is a Streamlit app. Three deployment options, ordered easiest → most control:

## Option 1: Streamlit Community Cloud (free, public, recommended)

1. Push this directory to a public GitHub repo:
   ```bash
   cd /Users/rr/nba-finals-2026
   git init && git add . && git commit -m "Initial dashboard"
   gh repo create RRGU26/nba-finals-2026 --public --source=. --push
   ```

2. Go to https://share.streamlit.io → "New app"
3. Connect your GitHub, select the repo, set:
   - Main file: `app.py`
   - Python version: 3.10+
4. Click Deploy. You'll get a URL like `https://nba-finals-2026.streamlit.app`

**Caveats:**
- Public — anyone with the URL can see it (no auth on free tier)
- Needs to be a public GitHub repo
- The Odds API key (used in `recommend_bet.py`) is in `nba-trader/.env` — if you want live odds in the deployed app, copy that key into the Streamlit Cloud **Secrets** panel and update the betting page to read from `st.secrets`

## Option 2: Run locally (no deployment)

```bash
cd /Users/rr/nba-finals-2026
uv run streamlit run app.py
```

Opens at http://localhost:8501. Stop with `Ctrl+C`.

To expose on your local network:
```bash
uv run streamlit run app.py --server.address 0.0.0.0
```
Then anyone on your wifi can hit `http://<your-mac-ip>:8501`.

## Option 3: Docker (for any cloud)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen
COPY . .
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

Build & run:
```bash
docker build -t nba-finals-2026 .
docker run -p 8501:8501 nba-finals-2026
```

Deploy to anywhere that runs Docker (Fly.io, Railway, Render, GCP Cloud Run, AWS ECS).

## Updating data after each Finals game

The dashboard reads from `data/team_games.parquet` and `logs/series_simulation.json`.
After each game completes:

```bash
cd /Users/rr/nba-finals-2026
uv run python fetch_data.py        # pulls new boxscore
uv run python autoresearch.py 400  # optional: re-train (takes ~5 min)
uv run python series_sim.py        # re-simulates remaining games
```

If deployed on Streamlit Cloud: push the updated parquet/json files to GitHub
and the app redeploys automatically. Alternative: add a `cron` job on your
local machine that pushes nightly.

## Auto-refresh setup (recommended)

Add to `~/.openclaw/cron/jobs.json` (or system cron):

```json
{
  "name": "nba-finals-refresh",
  "cron": "0 6 * * *",
  "command": "cd /Users/rr/nba-finals-2026 && uv run python fetch_data.py && uv run python series_sim.py"
}
```

Runs every morning at 6am, so the dashboard reflects yesterday's game when you wake up.
