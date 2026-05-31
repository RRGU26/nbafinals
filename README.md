# NBA Finals 2026 — Prediction Engine

Bayesian micro-model + live dashboard for **Knicks vs Spurs**.

Built specifically for this matchup — separate from the regular-season trader
because Finals dynamics (slower pace, tighter defense, shorter rotations) make
the season-trained model unreliable.

## What's inside

| File | Purpose |
|------|---------|
| `fetch_data.py` | Pull all NYK + SAS games (RS + playoffs) via nba_api |
| `features.py` | Recency-weighted, opponent-adjusted features |
| `model.py` | Bayesian linear regression with conjugate prior |
| `autoresearch.py` | Karpathy-style search: 400 experiments over hyperparams + feature subsets |
| `predict_game.py` | Per-game prediction (Analytic + Bayesian + Consensus) |
| `series_sim.py` | Monte Carlo simulator (10k trials) of the full 7-game series |
| `commentary.py` | Template-based prose generator |
| `recommend_bet.py` | Compare model edges to live Vegas lines |
| `app.py` | **Streamlit dashboard** — overview, predictions, betting, commentary, methodology |

## Quick start

```bash
cd /Users/rr/nba-finals-2026
uv sync
uv run python fetch_data.py       # one-time data pull (~30 sec)
uv run python autoresearch.py 400 # ~5 min — train model
uv run python series_sim.py       # simulate series
uv run streamlit run app.py       # launch dashboard at localhost:8501
```

## Current prediction (as of 2026-05-31)

- **NYK to win series: 74%**
- **Most likely outcome: NYK in 6 (29%)**
- **Expected series length: 5.7 games**

## Architecture

```
nba_api ─→ fetch_data.py ─→ data/*.parquet
                                  │
                                  ▼
                        features.py (recency-weighted)
                                  │
                                  ▼
                  ┌───────────────┴───────────────┐
                  ▼                                ▼
       model.py (Bayesian)              analytic_prediction()
                  ▼                                ▼
                  └────────────┬───────────────────┘
                               ▼
                       series_sim.py (Monte Carlo)
                               │
                               ▼
                           app.py (Streamlit)
```

## Deployment

See [DEPLOY.md](DEPLOY.md) — Streamlit Cloud (free, recommended), local, or Docker.

## Key limitations

- **Tiny sample**: only 16 playoff games per team. The Bayesian coefficients
  are heavily shrunk — the analytic formula does most of the work.
- **No player-level modeling**: predicts team outcomes only. Props are likely
  a better edge but require separate data.
- **Vegas usually wins**: when our model disagrees with Vegas by 5+ points,
  default to Vegas — they have injury and lineup info we don't.

See the dashboard's **Methodology** page for full details.
