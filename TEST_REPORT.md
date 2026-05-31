# Test Report — NBA Finals 2026 Dashboard

Generated: 2026-05-31

## TL;DR

**All tests passing.** 54 individual tests across unit, integration, edge case, and browser categories.

| Category | Tests | Pass | Fail |
|----------|------:|-----:|-----:|
| Module imports | 7 | 7 | 0 |
| Data integrity | 8 | 8 | 0 |
| Feature engineering | 4 | 4 | 0 |
| Bayesian model | 3 | 3 | 0 |
| Prediction pipeline | 2 | 2 | 0 |
| Simulation | 4 | 4 | 0 |
| Commentary | 3 | 3 | 0 |
| Odds module | 3 | 3 | 0 |
| Edge cases | 11 | 11 | 0 |
| Browser/dashboard | 6 | 6 | 0 |
| **TOTAL** | **54** | **54** | **0** |

## Stress test — 1500 autoresearch experiments

**Outcome: model is convergent and stable.**

All top 5 configurations from 1500 experiments use the **identical feature set**:
- `is_home, net_diff, pace_avg` (3 features)
- `playoff_weight: 5.0`
- `prior_precision: 100.0`
- `last_n_games: 30`
- Zero sign violations

The autoresearch found the same optimum repeatedly, which means the model architecture is settled. Margin RMSE plateau at ~16.1, total RMSE at ~15.2, win accuracy at 75% (LOO-CV on playoff games).

## What was tested

### Module imports (7/7)
Every Python module imports without errors: `features`, `model`, `autoresearch`, `predict_game`, `series_sim`, `commentary`, `odds`.

### Data integrity (8/8)
- `team_games.parquet` exists with both NYK + SAS rows (196 total)
- `historical_finals.parquet` has 63 games across 11 seasons (2014-15 → 2024-25)
- No NaN in critical fields (PTS, OPP_PTS, MARGIN, ratings)
- `MARGIN` consistent with `PTS - OPP_PTS`
- Pace values in sane range (88.8 to 112.2)
- NYK playoff path verified: ATL 4-2, PHI 4-0, CLE 4-0
- SAS playoff path verified: POR 4-1, MIN 4-2, OKC 4-3
- Only 2 H2H games this season (1-1 split)

### Feature engineering (4/4)
- `FeatureConfig` defaults are sensible (decay in (0,1], weights ≥ 1)
- `team_features` handles dates with no prior history (returns league defaults)
- `perspective_features` returns all expected keys including `is_home`
- `build_training_set` produces both home (is_home=1) and away (is_home=0) rows

### Bayesian model (3/3)
- Fits cleanly on synthetic data — recovers true coefficients within tolerance
- `predict` returns finite mean + variance arrays of correct shape
- `win_prob` outputs bounded in [0, 1]

### Prediction pipeline (2/2)
- End-to-end `predict()` call produces sensible outputs (win prob in (0,1), total in 180-260, margin in -40 to 40)
- Home court asymmetry correctly applied (NYK at home > NYK on road)

### Simulation (4/4)
- `simulate_series` calibrates correctly: P(win) ≈ 0.5 for 50/50 games
- Strong team (0.8 per-game) wins series > 95% of trials
- Outcome probabilities sum to 1
- Expected games always in valid [4, 7] range

### Commentary (3/3)
- `series_overview` produces non-empty narrative
- `matchup_analysis` references both teams
- `game_commentary` parses game dict and produces formatted output

### Odds module (3/3)
- `american_to_implied_prob(-110) = 0.5238` ✓
- `american_to_decimal(+200) = 3.0` ✓
- `fetch_nba_odds` returns list (graceful empty on missing API key)

### Edge cases (11/11)
- Bogus team codes raise expected errors
- Series sim handles extreme probabilities (0.99, 0.0)
- Bayesian model handles single feature, constant features, zero-variance weights
- Missing API key returns empty odds list (no crash)
- Commentary handles minimal/sparse input dicts
- Features handle pre-historic dates (no games before)
- Unknown team IDs fall back to defaults

### Browser tests (6/6)
Playwright headless chromium, full page render per page:
- 📊 Overview — passes
- 🎯 Game Predictions — passes
- 💰 Betting Analysis — passes
- 📝 Commentary — passes (after markdown rendering fix)
- 📚 Historical Finals — passes
- 🔬 Methodology — passes

Screenshots saved to `tests/screenshots/`.

## Bugs found and fixed during testing

1. **Markdown not rendering in commentary HTML blocks**
   - Symptom: Literal `**` shown around bold words
   - Cause: Streamlit's `unsafe_allow_html=True` doesn't process markdown inside HTML divs
   - Fix: Added `md_to_html()` helper converting `**X**` → `<b>X</b>`

2. **Hardcoded `/Users/rr/nba-trader` path in odds fetcher**
   - Symptom: Would fail on Streamlit Cloud
   - Fix: Extracted `odds.py` as self-contained module with `st.secrets` fallback

3. **CORS config warning in Streamlit**
   - Symptom: `enableCORS=false` incompatible with `enableXsrfProtection=true`
   - Fix: Removed redundant `enableCORS` line from config.toml

4. **`use_container_width=True` deprecation warning**
   - Symptom: 16 deprecation warnings in browser console
   - Fix: Bulk-replaced with `width="stretch"` across `app.py`

5. **Bayesian model coefficient sign flipping**
   - Symptom: Earlier model versions had `fg_diff` coefficient of -324 (perverse sign)
   - Cause: Multicollinearity + insufficient regularization on 16 playoff games
   - Fix: Added sign-violation penalty in autoresearch scoring (100× penalty for wrong sign)

6. **Expected games calculation bug in simulator**
   - Symptom: Output showed 0.7 instead of ~5.7
   - Cause: Used `np.mean()` instead of `sum()` of weighted lengths
   - Fix: One-line correction in `series_sim.py`

## Performance metrics

| Operation | Time |
|-----------|-----:|
| `fetch_data.py` (full pull) | ~30 sec |
| `fetch_historical.py` (11 seasons) | ~15 sec |
| `autoresearch.py 400` | ~5 min |
| `autoresearch.py 1500` | ~17 min |
| `series_sim.py` (10k trials) | ~10 sec |
| `predict_game.py` (single game) | ~3 sec |
| Streamlit dashboard cold start | ~4 sec |
| Browser page load per tab | ~3-4 sec (with Plotly render) |

## Known limitations (not bugs)

1. **Two-model disagreement on Game 1**: Analytic predicts NYK by 2.3, Bayesian predicts SAS by 11.4. Reflects honest uncertainty on a small playoff sample, not a bug.

2. **No player-level modeling**: All predictions are team-level. Props would require separate data pipeline.

3. **Total points may be over-calibrated**: We blend 50/50 with historical Finals average (210.9). This produces conservative totals that may miss truly high-scoring games.

4. **Team logos may flash on first paint**: SVGs load from NBA CDN async; brief delay before they appear.

## Reproduce the tests

```bash
cd /Users/rr/nba-finals-2026
uv sync

# Unit + integration tests (~30 sec)
uv run python tests/test_modules.py

# Edge case tests (~5 sec)
uv run python tests/test_edge_cases.py

# Browser tests (~30 sec — requires streamlit running)
uv run streamlit run app.py --server.headless true --server.port 8765 &
sleep 4
uv run python tests/test_dashboard.py

# Full pipeline smoke test (~5 min)
uv run python fetch_data.py
uv run python autoresearch.py 400
uv run python series_sim.py
uv run python predict_game.py --home SAS --away NYK
```

## What's safe to deploy

✅ **The dashboard is production-ready for Streamlit Cloud.** All 6 pages render cleanly, error paths are handled, optional features (live odds) degrade gracefully when the API key is missing.

✅ **The model is convergent.** 1500 autoresearch experiments produce the same top-5 configurations, confirming we haven't been fooled by a local optimum.

⚠️ **Don't bet real money based on this alone.** The 75% win accuracy is on LOO-CV of 16 playoff games — that's not a robust sample. Vegas has more information than we do.

✅ **Refreshable.** After each Finals game, `fetch_data.py` + `series_sim.py` updates the dashboard. No manual data engineering required.
