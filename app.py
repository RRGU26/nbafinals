"""NBA Finals 2026 Dashboard — Knicks vs Spurs prediction engine.

Run: uv run streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Finals 2026 · NYK vs SAS",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand colors
NYK_PRIMARY = "#006BB6"
NYK_ORANGE = "#F58426"
NYK_GRAD = "linear-gradient(135deg, #006BB6 0%, #00497F 100%)"
SAS_PRIMARY = "#000000"
SAS_SILVER = "#C4CED4"
SAS_GRAD = "linear-gradient(135deg, #1E1E1E 0%, #4A5258 100%)"

# Team logo CDN
NYK_LOGO = "https://cdn.nba.com/logos/nba/1610612752/global/L/logo.svg"
SAS_LOGO = "https://cdn.nba.com/logos/nba/1610612759/global/L/logo.svg"

# Plotly theme
PLOTLY_THEME = dict(
    layout=dict(
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", size=13, color="#1a1a1a"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=30, l=40, r=20),
        xaxis=dict(gridcolor="#e5e7eb", linecolor="#9ca3af", zerolinecolor="#9ca3af"),
        yaxis=dict(gridcolor="#e5e7eb", linecolor="#9ca3af", zerolinecolor="#9ca3af"),
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 1400px;}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #1f2937 50%, #0d1117 100%);
    padding: 2rem 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, rgba(245,132,38,0.15) 0%, transparent 60%);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    position: relative;
}
.hero-sub {
    font-size: 1.1rem;
    opacity: 0.7;
    margin: 0.25rem 0 0 0;
    position: relative;
}
.hero-stat {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.6;
}

/* Team cards */
.team-card {
    padding: 1.5rem;
    border-radius: 16px;
    color: white;
    position: relative;
    overflow: hidden;
    height: 100%;
    min-height: 280px;
}
.team-card-nyk { background: linear-gradient(135deg, #006BB6 0%, #00497F 60%, #F58426 100%); }
.team-card-sas { background: linear-gradient(135deg, #1f1f1f 0%, #4A5258 60%, #C4CED4 100%); }

.team-name {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.01em;
}
.team-record {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.25rem 0 0 0;
    letter-spacing: -0.04em;
}
.team-record-sub {
    font-size: 0.8rem;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.team-playoff {
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 1rem;
}
.team-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.2);
}
.team-stat {
    font-size: 0.85rem;
}
.team-stat-label {
    opacity: 0.7;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.team-stat-value {
    font-size: 1.1rem;
    font-weight: 600;
}

/* Pill / badge */
.pill {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.pill-blue { background: #dbeafe; color: #1e40af; }
.pill-green { background: #d1fae5; color: #065f46; }
.pill-red { background: #fee2e2; color: #991b1b; }
.pill-gray { background: #f3f4f6; color: #374151; }

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 2rem 0 0.5rem 0;
    letter-spacing: -0.015em;
    color: #1a1a1a;
}
.section-sub {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 1rem;
}

/* Commentary block */
.commentary-block {
    font-size: 1.05rem;
    line-height: 1.75;
    padding: 1.5rem 1.75rem;
    background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    border-radius: 12px;
    border-left: 4px solid #006BB6;
    margin: 1rem 0;
    color: #1f2937;
}
.commentary-block b { color: #006BB6; }

/* Bet rows */
.bet-row {
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: white;
}
.bet-row-edge {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border-left: 4px solid #10b981;
}
.bet-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
}
.bet-meta {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 0.25rem;
}
.bet-edge {
    font-size: 1.5rem;
    font-weight: 700;
    color: #059669;
}

/* Series sidebar bar */
.series-bar {
    display: flex;
    height: 36px;
    border-radius: 8px;
    overflow: hidden;
    margin: 0.5rem 0;
    font-weight: 600;
    color: white;
    font-size: 0.95rem;
    align-items: center;
}
.series-bar-nyk {
    background: linear-gradient(90deg, #006BB6 0%, #0085E0 100%);
    text-align: center;
    padding-right: 0.5rem;
}
.series-bar-sas {
    background: linear-gradient(90deg, #4A5258 0%, #1E1E1E 100%);
    text-align: center;
    padding-left: 0.5rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 0.8rem;
    padding-top: 3rem;
    margin-top: 4rem;
    border-top: 1px solid #e5e7eb;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
    background: #f9fafb;
}
.stTabs [aria-selected="true"] {
    background: #006BB6 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"
MODEL_DIR = Path(__file__).parent / "model"


@st.cache_data(ttl=3600)
def load_team_games() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "team_games.parquet")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


@st.cache_data(ttl=3600)
def load_historical_finals() -> pd.DataFrame | None:
    p = DATA_DIR / "historical_finals.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(ttl=600)
def load_series_sim() -> dict | None:
    p = LOG_DIR / "series_simulation.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(ttl=600)
def load_best_config_json() -> dict | None:
    p = MODEL_DIR / "best_config.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(ttl=600)
def load_experiments() -> pd.DataFrame | None:
    p = LOG_DIR / "experiments.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(ttl=86400)
def team_form_summary(team_name: str) -> dict:
    df = load_team_games()
    t = df[df["TEAM_NAME"] == team_name].sort_values("GAME_DATE")
    rs = t[t["SEASON_TYPE"] == "Regular Season"]
    po = t[t["SEASON_TYPE"] == "Playoffs"]
    return {
        "rs_wins": int((rs["WL"] == "W").sum()),
        "rs_losses": int((rs["WL"] == "L").sum()),
        "po_wins": int((po["WL"] == "W").sum()),
        "po_losses": int((po["WL"] == "L").sum()),
        "rs_off_rating": float(rs["OFF_RATING"].mean()),
        "rs_def_rating": float(rs["DEF_RATING"].mean()),
        "rs_pace": float(rs["PACE"].mean()),
        "po_off_rating": float(po["OFF_RATING"].mean()),
        "po_def_rating": float(po["DEF_RATING"].mean()),
        "po_pace": float(po["PACE"].mean()),
        "last_5_margin": float(t.tail(5)["MARGIN"].mean()) if len(t) >= 5 else 0.0,
        "po_games": po.sort_values("GAME_DATE"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2rem;">🏀</div>
        <div style="font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em;">FINALS 2026</div>
        <div style="font-size: 0.85rem; color: #6b7280;">Prediction Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    PAGES = {
        "📊 Overview": "overview",
        "🎯 Game Predictions": "games",
        "💰 Betting Analysis": "betting",
        "📝 Commentary": "commentary",
        "📚 Historical Finals": "historical",
        "🔬 Methodology": "methodology",
    }
    # Use session_state so top-of-page nav can sync with sidebar
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = list(PAGES.keys())[0]
    page = st.radio("Page", list(PAGES.keys()),
                     index=list(PAGES.keys()).index(st.session_state.selected_page),
                     label_visibility="collapsed",
                     key="sidebar_nav")
    st.session_state.selected_page = page

    st.markdown("---")

    sim = load_series_sim()
    if sim:
        p_nyk = sim["simulation"]["p_nyk_wins"]
        p_sas = 1 - p_nyk
        st.markdown('<div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7280; font-weight: 600;">Series Odds</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="series-bar">
            <div class="series-bar-nyk" style="width: {p_nyk*100}%;">NYK {p_nyk:.0%}</div>
            <div class="series-bar-sas" style="width: {p_sas*100}%;">SAS {p_sas:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
        most_likely = max(sim["simulation"]["outcomes"].items(), key=lambda x: x[1])
        st.markdown(f'<div style="font-size: 0.85rem; color: #6b7280;">Most likely: <b style="color: #1a1a1a;">{most_likely[0]}</b> ({most_likely[1]:.0%})</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.85rem; color: #6b7280;">Expected games: <b style="color: #1a1a1a;">{sim["simulation"]["expected_games"]:.1f}</b></div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("⚙️ Update commands"):
        st.code("uv run python fetch_data.py", language="bash")
        st.code("uv run python autoresearch.py 400", language="bash")
        st.code("uv run python series_sim.py", language="bash")

    # Data freshness
    from datetime import datetime
    try:
        latest_game_date = load_team_games()["GAME_DATE"].max()
        days_old = (datetime.now() - latest_game_date).days
        freshness_color = "#10b981" if days_old <= 2 else ("#fbbf24" if days_old <= 7 else "#ef4444")
        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #6b7280; margin-top: 1.5rem; padding: 0.5rem; background: #f9fafb; border-radius: 6px;">
            <div style="font-weight: 600; color: #374151;">Data Freshness</div>
            <div style="margin-top: 0.25rem;">
                Last game: <b>{latest_game_date.strftime('%b %d')}</b>
                <span style="color: {freshness_color};">●</span>
                {days_old}d ago
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown('<div style="font-size: 0.7rem; color: #9ca3af; margin-top: 1.5rem;">Built with Streamlit · Not financial advice</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP NAV (visible even when sidebar is collapsed — mobile fallback)
# ─────────────────────────────────────────────────────────────────────────────
nav_cols = st.columns(len(PAGES))
for i, p in enumerate(PAGES.keys()):
    with nav_cols[i]:
        is_active = (p == page)
        btn_type = "primary" if is_active else "secondary"
        if st.button(p, key=f"topnav_{i}", type=btn_type, width="stretch"):
            st.session_state.selected_page = p
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if PAGES[page] == "overview":
    # Hero
    if sim:
        p_nyk = sim["simulation"]["p_nyk_wins"]
        most_likely = max(sim["simulation"]["outcomes"].items(), key=lambda x: x[1])
        favorite = "Knicks" if p_nyk > 0.5 else "Spurs"
        fav_prob = max(p_nyk, 1 - p_nyk)
        st.markdown(f"""
        <div class="hero">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p class="hero-title">NBA Finals 2026</p>
                    <p class="hero-sub">New York Knicks vs San Antonio Spurs</p>
                </div>
                <div style="text-align: right;">
                    <div class="hero-stat">Model Pick</div>
                    <div style="font-size: 2rem; font-weight: 800; line-height: 1; margin-top: 0.2rem;">{favorite} {fav_prob:.0%}</div>
                    <div style="font-size: 0.95rem; opacity: 0.85;">Most likely: {most_likely[0]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero">
            <p class="hero-title">NBA Finals 2026</p>
            <p class="hero-sub">New York Knicks vs San Antonio Spurs</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("No simulation yet. Run `uv run python series_sim.py`.")
        st.stop()

    # Team cards
    nyk = team_form_summary("New York Knicks")
    sas = team_form_summary("San Antonio Spurs")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(f"""
        <div class="team-card team-card-nyk">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <p class="team-name">🗽 New York Knicks</p>
                    <p class="team-record">{nyk['rs_wins']}-{nyk['rs_losses']}</p>
                    <p class="team-record-sub">Regular Season</p>
                </div>
                <img src="{NYK_LOGO}" style="height: 80px; opacity: 0.85;">
            </div>
            <div class="team-playoff">Playoffs: <b>{nyk['po_wins']}-{nyk['po_losses']}</b></div>
            <div class="team-stat-grid">
                <div class="team-stat">
                    <div class="team-stat-label">Playoff ORtg</div>
                    <div class="team-stat-value">{nyk['po_off_rating']:.1f}</div>
                </div>
                <div class="team-stat">
                    <div class="team-stat-label">Playoff DRtg</div>
                    <div class="team-stat-value">{nyk['po_def_rating']:.1f}</div>
                </div>
                <div class="team-stat">
                    <div class="team-stat-label">Playoff Net</div>
                    <div class="team-stat-value">{nyk['po_off_rating']-nyk['po_def_rating']:+.1f}</div>
                </div>
                <div class="team-stat">
                    <div class="team-stat-label">L5 Margin</div>
                    <div class="team-stat-value">{nyk['last_5_margin']:+.1f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="team-card team-card-sas">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <p class="team-name">🤠 San Antonio Spurs</p>
                    <p class="team-record">{sas['rs_wins']}-{sas['rs_losses']}</p>
                    <p class="team-record-sub">Regular Season</p>
                </div>
                <img src="{SAS_LOGO}" style="height: 80px; opacity: 0.85;">
            </div>
            <div class="team-playoff">Playoffs: <b>{sas['po_wins']}-{sas['po_losses']}</b></div>
            <div class="team-stat-grid">
                <div class="team-stat">
                    <div class="team-stat-label">Playoff ORtg</div>
                    <div class="team-stat-value">{sas['po_off_rating']:.1f}</div>
                </div>
                <div class="team-stat">
                    <div class="team-stat-label">Playoff DRtg</div>
                    <div class="team-stat-value">{sas['po_def_rating']:.1f}</div>
                </div>
                <div class="team-stat">
                    <div class="team-stat-label">Playoff Net</div>
                    <div class="team-stat-value">{sas['po_off_rating']-sas['po_def_rating']:+.1f}</div>
                </div>
                <div class="team-stat">
                    <div class="team-stat-label">L5 Margin</div>
                    <div class="team-stat-value">{sas['last_5_margin']:+.1f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Radar chart
    st.markdown('<p class="section-header">📊 Team Comparison Radar</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Playoff stats normalized to percentile of NBA scale</p>', unsafe_allow_html=True)

    categories = ["Offense<br>Rating", "Defense<br>Rating", "Pace", "Net<br>Rating", "Recent<br>Form"]

    def normalize(val, lo, hi):
        return min(100, max(0, (val - lo) / (hi - lo) * 100))

    def def_normalize(val, lo, hi):
        # Lower is better for defense — invert
        return 100 - normalize(val, lo, hi)

    nyk_radar = [
        normalize(nyk['po_off_rating'], 105, 125),
        def_normalize(nyk['po_def_rating'], 95, 115),
        normalize(nyk['po_pace'], 92, 105),
        normalize(nyk['po_off_rating']-nyk['po_def_rating'], -10, 25),
        normalize(nyk['last_5_margin'], -15, 25),
    ]
    sas_radar = [
        normalize(sas['po_off_rating'], 105, 125),
        def_normalize(sas['po_def_rating'], 95, 115),
        normalize(sas['po_pace'], 92, 105),
        normalize(sas['po_off_rating']-sas['po_def_rating'], -10, 25),
        normalize(sas['last_5_margin'], -15, 25),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=nyk_radar + [nyk_radar[0]], theta=categories + [categories[0]],
                                    fill="toself", name="Knicks",
                                    line_color=NYK_PRIMARY, fillcolor="rgba(0, 107, 182, 0.25)"))
    fig.add_trace(go.Scatterpolar(r=sas_radar + [sas_radar[0]], theta=categories + [categories[0]],
                                    fill="toself", name="Spurs",
                                    line_color="#4A5258", fillcolor="rgba(74, 82, 88, 0.25)"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#e5e7eb"),
                     angularaxis=dict(gridcolor="#e5e7eb")),
        showlegend=True,
        height=420,
        margin=dict(t=20, b=20, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13),
    )
    st.plotly_chart(fig, width="stretch")

    # Series outcome distribution
    st.markdown('<p class="section-header">📈 Series Outcome Distribution</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Monte Carlo simulation, 10,000 trials</p>', unsafe_allow_html=True)

    outcomes = sim["simulation"]["outcomes"]
    nyk_o = [(k, v) for k, v in outcomes.items() if "NYK" in k]
    sas_o = [(k, v) for k, v in outcomes.items() if "SAS" in k]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[k for k, v in nyk_o], y=[v for k, v in nyk_o], name="Knicks",
                          marker_color=NYK_PRIMARY,
                          text=[f"{v:.0%}" for k, v in nyk_o], textposition="outside",
                          textfont=dict(family="Inter", size=12, color="#1a1a1a")))
    fig.add_trace(go.Bar(x=[k for k, v in sas_o], y=[v for k, v in sas_o], name="Spurs",
                          marker_color="#4A5258",
                          text=[f"{v:.0%}" for k, v in sas_o], textposition="outside",
                          textfont=dict(family="Inter", size=12, color="#1a1a1a")))
    fig.update_layout(height=380, yaxis_tickformat=".0%", yaxis_title="",
                      barmode="group", margin=dict(t=30, b=30),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=12))
    fig.update_yaxes(range=[0, max([v for k, v in outcomes.items()]) * 1.2])
    st.plotly_chart(fig, width="stretch")

    # Playoff path timeline
    st.markdown('<p class="section-header">🛤️ Playoff Path to Finals</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")

    def render_path(games, color, label):
        fig = go.Figure()
        games_sorted = games.sort_values("GAME_DATE")
        colors = [color if w == "W" else "#dc2626" for w in games_sorted["WL"]]
        fig.add_trace(go.Bar(
            x=list(range(1, len(games_sorted)+1)),
            y=games_sorted["MARGIN"],
            marker_color=colors,
            text=games_sorted.apply(lambda r: f"{r['PTS']}-{r['OPP_PTS']}", axis=1),
            textposition="outside",
            hovertext=games_sorted.apply(
                lambda r: f"vs {r['OPP_TEAM_ABBREVIATION']}<br>{r['GAME_DATE'].strftime('%b %d')}<br>{r['WL']} {r['PTS']}-{r['OPP_PTS']}", axis=1),
            hoverinfo="text",
        ))
        fig.add_hline(y=0, line_color="#374151")
        fig.update_layout(height=300, title=label, showlegend=False,
                          xaxis_title="Playoff Game #", yaxis_title="Margin",
                          margin=dict(t=40, b=30), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter, sans-serif", size=11))
        return fig

    with col1:
        st.plotly_chart(render_path(nyk["po_games"], NYK_PRIMARY, "Knicks: 12-2 in playoffs"),
                          width="stretch")
    with col2:
        st.plotly_chart(render_path(sas["po_games"], "#4A5258", "Spurs: 12-6 in playoffs"),
                          width="stretch")

    # H2H
    st.markdown('<p class="section-header">🤝 Head-to-Head This Season</p>', unsafe_allow_html=True)
    tg = load_team_games()
    nyk_games = tg[tg["TEAM_NAME"] == "New York Knicks"]
    sas_games = tg[tg["TEAM_NAME"] == "San Antonio Spurs"]
    h2h_ids = set(nyk_games["GAME_ID"]) & set(sas_games["GAME_ID"])
    h2h = nyk_games[nyk_games["GAME_ID"].isin(h2h_ids)].sort_values("GAME_DATE")

    if len(h2h) > 0:
        col1, col2 = st.columns([1, 1])
        for idx, (i, g) in enumerate(h2h.iterrows()):
            col = col1 if idx % 2 == 0 else col2
            location = "vs" if "vs" in g["MATCHUP"] else "@"
            with col:
                bg = "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)" if g["WL"] == "W" else "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
                border = "#10b981" if g["WL"] == "W" else "#ef4444"
                st.markdown(f"""
                <div style="padding: 1rem 1.25rem; background: {bg}; border-radius: 10px; border-left: 4px solid {border}; margin-bottom: 0.5rem;">
                    <div style="font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;">{g['GAME_DATE'].strftime('%B %d, %Y')}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; margin-top: 0.2rem;">
                        NYK {location} SAS · <span style="color: {border};">{g['WL']}</span>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #111827;">{g['PTS']}–{g['OPP_PTS']}</div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GAME PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
elif PAGES[page] == "games":
    st.markdown('<p class="hero-title" style="color:#1a1a1a; font-size: 2.2rem;">🎯 Game Predictions</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280; font-size: 1rem;">Per-game win probability, margin, and total — based on consensus model</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not sim:
        st.warning("Run `uv run python series_sim.py`.")
        st.stop()

    per_game = sim["per_game"]

    # Hero stat row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div style="background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb;"><div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">Series Prediction</div><div style="font-size: 1.8rem; font-weight: 800; color: #006BB6;">NYK {sim["simulation"]["p_nyk_wins"]:.0%}</div></div>', unsafe_allow_html=True)
    with col2:
        ml = max(sim["simulation"]["outcomes"].items(), key=lambda x: x[1])
        st.markdown(f'<div style="background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb;"><div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">Most Likely</div><div style="font-size: 1.8rem; font-weight: 800; color: #111827;">{ml[0]}</div><div style="font-size: 0.85rem; color: #6b7280;">{ml[1]:.0%} probability</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div style="background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb;"><div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">Expected Games</div><div style="font-size: 1.8rem; font-weight: 800; color: #111827;">{sim["simulation"]["expected_games"]:.1f}</div></div>', unsafe_allow_html=True)
    with col4:
        avg_total = sum(p["predicted_total"] for p in per_game) / len(per_game)
        st.markdown(f'<div style="background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb;"><div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">Avg Total</div><div style="font-size: 1.8rem; font-weight: 800; color: #111827;">{avg_total:.0f}</div></div>', unsafe_allow_html=True)

    # Win probability chart
    st.markdown('<p class="section-header">Per-game win probability</p>', unsafe_allow_html=True)
    games = [f"G{p['game']}<br>{p['away']} @ {p['home']}" for p in per_game]
    nyk_probs = [p["nyk_win_prob"] for p in per_game]
    sas_probs = [1 - p for p in nyk_probs]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=games, y=nyk_probs, name="NYK win", marker_color=NYK_PRIMARY,
                          text=[f"{p:.0%}" for p in nyk_probs], textposition="inside",
                          textfont=dict(color="white", size=14)))
    fig.add_trace(go.Bar(x=games, y=sas_probs, name="SAS win", marker_color="#4A5258",
                          text=[f"{p:.0%}" for p in sas_probs], textposition="inside",
                          textfont=dict(color="white", size=14)))
    fig.update_layout(barmode="stack", yaxis_tickformat=".0%", height=400,
                      yaxis_title="", margin=dict(t=20, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#9ca3af")
    st.plotly_chart(fig, width="stretch")

    # Margin & total side by side
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<p class="section-header">Predicted margin (home view)</p>', unsafe_allow_html=True)
        margins = [p["predicted_margin_home"] for p in per_game]
        colors = [NYK_PRIMARY if p["home"] == "NYK" else "#4A5258" for p in per_game]
        fig = go.Figure(go.Bar(x=games, y=margins, marker_color=colors,
                                text=[f"{m:+.1f}" for m in margins], textposition="outside",
                                textfont=dict(family="Inter", size=12)))
        fig.update_layout(height=350, yaxis_title="", margin=dict(t=20, b=20),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", size=11))
        fig.add_hline(y=0, line_color="#374151")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.markdown('<p class="section-header">Predicted total points</p>', unsafe_allow_html=True)
        totals = [p["predicted_total"] for p in per_game]
        fig = go.Figure(go.Bar(x=games, y=totals, marker_color="#F58426",
                                text=[f"{t:.0f}" for t in totals], textposition="outside",
                                textfont=dict(family="Inter", size=12)))
        fig.update_layout(height=350, yaxis_title="",
                          yaxis_range=[min(totals)-15, max(totals)+15], margin=dict(t=20, b=20),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", size=11))
        st.plotly_chart(fig, width="stretch")

    # Detailed table
    st.markdown('<p class="section-header">Game-by-game detail</p>', unsafe_allow_html=True)
    df = pd.DataFrame([{
        "Game": p["game"],
        "Matchup": f"{p['away']} @ {p['home']}",
        "Home Win%": f"{p['home_win_prob']:.1%}",
        "NYK Win%": f"{p['nyk_win_prob']:.1%}",
        "Margin (home)": f"{p['predicted_margin_home']:+.1f}",
        "Total": f"{p['predicted_total']:.1f}",
        "Analytic": f"{p['analytic_margin_home']:+.1f}",
        "Bayesian": f"{p['bayes_margin_home']:+.1f}",
    } for p in per_game])
    st.dataframe(df, width="stretch", hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BETTING
# ─────────────────────────────────────────────────────────────────────────────
elif PAGES[page] == "betting":
    st.markdown('<p class="hero-title" style="color:#1a1a1a; font-size: 2.2rem;">💰 Betting Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280;">Model probabilities vs live Vegas lines · ¼-Kelly sizing capped at 5%</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        refresh = st.button("🔄 Refresh odds", type="primary", width="stretch")
    odds_cache_path = LOG_DIR / "last_odds_snapshot.json"

    if refresh or not odds_cache_path.exists():
        try:
            from odds import fetch_nba_odds, find_game
            with st.spinner("Fetching live Finals odds..."):
                odds_raw = fetch_nba_odds()
                if not odds_raw:
                    st.warning("No odds available (set ODDS_API_KEY env var or st.secrets).")
                else:
                    game = find_game(odds_raw, "Spurs", "Knicks") or find_game(odds_raw, "Knicks", "Spurs")
                    if game:
                        with open(odds_cache_path, "w") as f:
                            json.dump(game, f, indent=2, default=str)
                        st.success("Odds refreshed", icon="✅")
                    else:
                        st.info("Knicks-Spurs game not found in current NBA odds slate.")
        except Exception as e:
            st.error(f"Failed to fetch odds: {e}")

    if not odds_cache_path.exists():
        st.warning("No odds yet. Click 'Refresh odds'.")
        st.stop()

    with open(odds_cache_path) as f:
        game = json.load(f)

    home_full = game["home_team"]
    away_full = game["away_team"]

    def best_price(market_key, side_filter):
        best = None
        for bk in game["bookmakers"]:
            for m in bk["markets"]:
                if m["key"] != market_key: continue
                for o in m["outcomes"]:
                    if not side_filter(o): continue
                    if best is None or o["price"] > best["price"]:
                        best = {**o, "book": bk["title"]}
        return best

    def american_to_implied(odds):
        return abs(odds)/(abs(odds)+100) if odds < 0 else 100/(odds+100)

    def american_to_decimal(odds):
        return 1 + (odds/100 if odds > 0 else 100/abs(odds))

    ml_home = best_price("h2h", lambda o: o["name"] == home_full)
    ml_away = best_price("h2h", lambda o: o["name"] == away_full)
    sp_home = best_price("spreads", lambda o: o["name"] == home_full)
    sp_away = best_price("spreads", lambda o: o["name"] == away_full)
    over = best_price("totals", lambda o: o["name"] == "Over")
    under = best_price("totals", lambda o: o["name"] == "Under")

    st.markdown(f'<p class="section-header">{away_full} @ {home_full}</p>', unsafe_allow_html=True)

    if not sim:
        st.error("No simulation. Run `series_sim.py`.")
        st.stop()

    g1 = sim["per_game"][0]
    home_short = "SAS" if "Spurs" in home_full else "NYK"
    away_short = "NYK" if home_short == "SAS" else "SAS"

    model_home_winp = g1["home_win_prob"]
    model_margin = g1["predicted_margin_home"]
    model_total = g1["predicted_total"]

    from scipy.stats import norm
    MARGIN_STD = 14.3
    TOTAL_STD = 17.0

    rows = []
    if ml_home:
        edge = model_home_winp - american_to_implied(ml_home["price"])
        rows.append({"name": f"{home_short} ML", "odds": ml_home["price"], "book": ml_home["book"],
                     "model_p": model_home_winp, "implied": american_to_implied(ml_home["price"]),
                     "edge": edge})
    if ml_away:
        away_p = 1 - model_home_winp
        edge = away_p - american_to_implied(ml_away["price"])
        rows.append({"name": f"{away_short} ML", "odds": ml_away["price"], "book": ml_away["book"],
                     "model_p": away_p, "implied": american_to_implied(ml_away["price"]),
                     "edge": edge})
    if sp_home:
        p_home_cover = float(norm.cdf((model_margin - (-sp_home["point"])) / MARGIN_STD))
        rows.append({"name": f"{home_short} {sp_home['point']:+.1f}", "odds": sp_home["price"],
                     "book": sp_home["book"], "model_p": p_home_cover,
                     "implied": american_to_implied(sp_home["price"]),
                     "edge": p_home_cover - american_to_implied(sp_home["price"])})
    if sp_away:
        p_away_cover = 1 - float(norm.cdf((model_margin - (-sp_home["point"])) / MARGIN_STD))
        rows.append({"name": f"{away_short} {sp_away['point']:+.1f}", "odds": sp_away["price"],
                     "book": sp_away["book"], "model_p": p_away_cover,
                     "implied": american_to_implied(sp_away["price"]),
                     "edge": p_away_cover - american_to_implied(sp_away["price"])})
    if over:
        p_over = float(norm.cdf((model_total - over["point"]) / TOTAL_STD))
        rows.append({"name": f"Over {over['point']}", "odds": over["price"], "book": over["book"],
                     "model_p": p_over, "implied": american_to_implied(over["price"]),
                     "edge": p_over - american_to_implied(over["price"])})
    if under:
        p_under = 1 - float(norm.cdf((model_total - under["point"]) / TOTAL_STD))
        rows.append({"name": f"Under {under['point']}", "odds": under["price"], "book": under["book"],
                     "model_p": p_under, "implied": american_to_implied(under["price"]),
                     "edge": p_under - american_to_implied(under["price"])})

    # Edges chart
    st.markdown('<p class="section-sub">Sorted by edge (model probability − implied probability)</p>', unsafe_allow_html=True)
    rows_sorted = sorted(rows, key=lambda x: -x["edge"])
    fig = go.Figure(go.Bar(
        x=[r["edge"] for r in rows_sorted],
        y=[r["name"] for r in rows_sorted],
        orientation="h",
        marker_color=["#10b981" if r["edge"] > 0.03 else ("#fbbf24" if r["edge"] > 0 else "#ef4444") for r in rows_sorted],
        text=[f"{r['edge']:+.1%}" for r in rows_sorted],
        textposition="outside",
        textfont=dict(family="Inter", size=12),
    ))
    fig.add_vline(x=0.03, line_dash="dash", line_color="#10b981")
    fig.add_vline(x=0, line_color="#374151")
    fig.update_layout(height=300, xaxis_tickformat=".0%", margin=dict(t=20, b=30, l=120, r=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", size=12))
    st.plotly_chart(fig, width="stretch")

    # Recommendations
    st.markdown('<p class="section-header">💡 Recommended Bets</p>', unsafe_allow_html=True)
    profitable = [r for r in rows if r["edge"] > 0.03]
    if not profitable:
        st.info("**No bets clear the 3% edge threshold.** Skip this slate. The model's noise exceeds its edge here.", icon="🚫")
    else:
        for r in sorted(profitable, key=lambda x: -x["edge"]):
            decimal = american_to_decimal(r["odds"])
            model_p = r["model_p"]
            b = decimal - 1
            kelly = max(0, min((model_p * b - (1 - model_p)) / b * 0.25, 0.05))
            st.markdown(f"""
            <div class="bet-row bet-row-edge">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="bet-name">{r['name']}</div>
                        <div class="bet-meta">{r['odds']:+d} at {r['book']} · Model: {r['model_p']:.1%} · Implied: {r['implied']:.1%}</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="bet-edge">+{r['edge']*100:.1f}%</div>
                        <div class="bet-meta">Kelly {kelly*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Full table
    with st.expander("📋 All markets detail"):
        df = pd.DataFrame([{
            "Bet": r["name"], "Odds": f"{r['odds']:+d}", "Book": r["book"],
            "Model": f"{r['model_p']:.1%}", "Implied": f"{r['implied']:.1%}",
            "Edge": f"{r['edge']:+.1%}"
        } for r in rows])
        st.dataframe(df, width="stretch", hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: COMMENTARY
# ─────────────────────────────────────────────────────────────────────────────
elif PAGES[page] == "commentary":
    st.markdown('<p class="hero-title" style="color:#1a1a1a; font-size: 2.2rem;">📝 Series Commentary</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280;">Auto-generated narrative from model outputs</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not sim:
        st.warning("Run `uv run python series_sim.py`.")
        st.stop()

    from commentary import series_overview, matchup_analysis, game_commentary
    from features import FeatureConfig, load_data, league_baselines, team_features
    from predict_game import load_best_config, TEAM_IDS

    config, _, _, _ = load_best_config()
    team_games, _ = load_data()
    baselines = league_baselines(team_games)
    as_of = pd.Timestamp("2026-06-04")
    sas_form = team_features(TEAM_IDS["SAS"], as_of, team_games, baselines, config)
    nyk_form = team_features(TEAM_IDS["NYK"], as_of, team_games, baselines, config)

    tab1, tab2, tab3 = st.tabs(["📰 Series Outlook", "⚔️ Matchup Analysis", "🎮 Game Previews"])

    def md_to_html(text: str) -> str:
        """Convert **bold** to <b>bold</b> for HTML rendering."""
        import re
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    with tab1:
        st.markdown(f'<div class="commentary-block">{md_to_html(series_overview(sim))}</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown(f'<div class="commentary-block">{md_to_html(matchup_analysis(nyk_form, sas_form, "Knicks", "Spurs"))}</div>', unsafe_allow_html=True)

    with tab3:
        # Show only unique games (G1+G2 same prediction; just show distinct combos)
        seen = set()
        for g in sim["per_game"]:
            key = f"{g['home']}_{g['away']}"
            if key in seen:
                continue
            seen.add(key)
            st.markdown(f'<div class="commentary-block">{md_to_html(game_commentary(g))}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HISTORICAL FINALS
# ─────────────────────────────────────────────────────────────────────────────
elif PAGES[page] == "historical":
    st.markdown('<p class="hero-title" style="color:#1a1a1a; font-size: 2.2rem;">📚 Historical NBA Finals</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280;">11 seasons of Finals data (2015–2025) used to calibrate the model</p>', unsafe_allow_html=True)
    st.markdown("---")

    hist = load_historical_finals()
    if hist is None:
        st.warning("Run `uv run python fetch_historical.py` first.")
        st.stop()

    games = hist.groupby("GAME_ID").first()

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Avg Total Pts", f"{games['TOTAL_POINTS'].mean():.1f}", "vs ~228 in regular season"),
        ("Avg Pace", f"{hist['PACE'].mean():.1f}", "vs ~99 in regular season"),
        ("Home Win%", f"{(hist[hist['IS_HOME']]['WL']=='W').mean():.0%}", "Home court advantage"),
        ("Blowout %", f"{(hist['MARGIN'].abs()>15).mean():.0%}", "Games decided by 15+"),
    ]
    for i, (label, val, sub) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.markdown(f'<div style="background: white; padding: 1.2rem; border-radius: 12px; border: 1px solid #e5e7eb;"><div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em;">{label}</div><div style="font-size: 2rem; font-weight: 800; color: #111827; line-height: 1;">{val}</div><div style="font-size: 0.8rem; color: #9ca3af; margin-top: 0.4rem;">{sub}</div></div>', unsafe_allow_html=True)

    # Per-season chart
    st.markdown('<p class="section-header">📊 Per-season trends</p>', unsafe_allow_html=True)
    by_season = hist.groupby("SEASON").agg(
        avg_total=("TOTAL_POINTS", "mean"),
        avg_pace=("PACE", "mean"),
        games=("GAME_ID", "nunique"),
    ).reset_index()

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average Total Points", "Average Pace"))
    fig.add_trace(go.Scatter(x=by_season["SEASON"], y=by_season["avg_total"], mode="lines+markers",
                              marker=dict(size=10, color=NYK_PRIMARY), line=dict(width=3, color=NYK_PRIMARY)),
                   row=1, col=1)
    fig.add_trace(go.Scatter(x=by_season["SEASON"], y=by_season["avg_pace"], mode="lines+markers",
                              marker=dict(size=10, color="#F58426"), line=dict(width=3, color="#F58426")),
                   row=1, col=2)
    fig.update_layout(height=350, showlegend=False, margin=dict(t=40, b=30),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", size=11))
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    # Distribution
    st.markdown('<p class="section-header">📈 Total points distribution</p>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=games["TOTAL_POINTS"], nbinsx=20, marker_color=NYK_PRIMARY,
                                 opacity=0.85, name="Historical Finals games"))
    fig.add_vline(x=games["TOTAL_POINTS"].mean(), line_dash="dash", line_color="#F58426",
                   annotation_text=f"Mean: {games['TOTAL_POINTS'].mean():.0f}")
    fig.update_layout(height=300, xaxis_title="Total points", yaxis_title="Games",
                      margin=dict(t=20, b=30), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", size=11))
    st.plotly_chart(fig, width="stretch")

    # Recent results
    with st.expander("📋 All 63 historical Finals games"):
        display = games[["SEASON", "TEAM_ABBREVIATION", "OPP_TEAM_ABBREVIATION", "PTS", "OPP_PTS",
                          "TOTAL_POINTS", "MARGIN", "PACE"]].copy()
        display.columns = ["Season", "Team", "Opp", "PTS", "OPP_PTS", "Total", "Margin", "Pace"]
        st.dataframe(display.round(1), width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
elif PAGES[page] == "methodology":
    st.markdown('<p class="hero-title" style="color:#1a1a1a; font-size: 2.2rem;">🔬 Methodology</p>', unsafe_allow_html=True)
    st.markdown("---")

    from commentary import methodology_blurb
    st.markdown(methodology_blurb())

    st.markdown("---")
    st.markdown('<p class="section-header">Autoresearch results</p>', unsafe_allow_html=True)
    exp = load_experiments()
    if exp is not None and len(exp) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div style="background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb;"><div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">Experiments</div><div style="font-size: 2rem; font-weight: 800; color: #111827;">{len(exp)}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div style="background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb;"><div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">Best Margin RMSE</div><div style="font-size: 2rem; font-weight: 800; color: #111827;">{exp["margin_rmse"].min():.1f}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div style="background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb;"><div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase;">Best Win Accuracy</div><div style="font-size: 2rem; font-weight: 800; color: #10b981;">{exp["win_accuracy"].max():.0%}</div></div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">RMSE distribution across experiments</p>', unsafe_allow_html=True)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Margin RMSE", "Total RMSE"))
        fig.add_trace(go.Histogram(x=exp["margin_rmse"], marker_color=NYK_PRIMARY, nbinsx=25), row=1, col=1)
        fig.add_trace(go.Histogram(x=exp["total_rmse"], marker_color="#F58426", nbinsx=25), row=1, col=2)
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=20),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", size=11))
        st.plotly_chart(fig, width="stretch")

        st.markdown('<p class="section-header">Top 10 configurations</p>', unsafe_allow_html=True)
        top10 = exp.nsmallest(10, "combined_score")[["margin_rmse", "total_rmse", "win_accuracy",
                                                       "recency_decay", "playoff_weight", "use_opp_adjustment",
                                                       "prior_precision", "n_features", "feature_cols"]]
        st.dataframe(top10, width="stretch", hide_index=True)

    st.markdown('<p class="section-header">Current best config</p>', unsafe_allow_html=True)
    best = load_best_config_json()
    if best:
        col1, col2 = st.columns(2)
        with col1:
            cfg = {k: v for k, v in best.items() if k != "feature_cols"}
            st.json(cfg)
        with col2:
            st.markdown("**Selected features:**")
            for f in best["feature_cols"].split(","):
                st.markdown(f"- `{f}`")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    NBA Finals 2026 Prediction Engine · Built with Streamlit ·
    <a href="https://github.com/RRGU26/nbafinals" target="_blank">View source on GitHub</a> ·
    Not financial advice
</div>
""", unsafe_allow_html=True)
