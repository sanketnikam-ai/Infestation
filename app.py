"""
🦟 Mosquito Google Trends – India State Analysis
Streamlit App | Deployable via GitHub → Streamlit Cloud
"""

import time
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="🦟 Mosquito Trends India",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
KEYWORD = "Mosquito"

INDIAN_STATES = {
    "Andhra Pradesh": "IN-AP",
    "Arunachal Pradesh": "IN-AR",
    "Assam": "IN-AS",
    "Bihar": "IN-BR",
    "Chhattisgarh": "IN-CT",
    "Goa": "IN-GA",
    "Gujarat": "IN-GJ",
    "Haryana": "IN-HR",
    "Himachal Pradesh": "IN-HP",
    "Jharkhand": "IN-JH",
    "Karnataka": "IN-KA",
    "Kerala": "IN-KL",
    "Madhya Pradesh": "IN-MP",
    "Maharashtra": "IN-MH",
    "Manipur": "IN-MN",
    "Meghalaya": "IN-ML",
    "Mizoram": "IN-MZ",
    "Nagaland": "IN-NL",
    "Odisha": "IN-OR",
    "Punjab": "IN-PB",
    "Rajasthan": "IN-RJ",
    "Sikkim": "IN-SK",
    "Tamil Nadu": "IN-TN",
    "Telangana": "IN-TG",
    "Tripura": "IN-TR",
    "Uttar Pradesh": "IN-UP",
    "Uttarakhand": "IN-UT",
    "West Bengal": "IN-WB",
    "Delhi": "IN-DL",
    "Jammu & Kashmir": "IN-JK",
    "Ladakh": "IN-LA",
    "Chandigarh": "IN-CH",
    "Puducherry": "IN-PY",
    "Andaman & Nicobar": "IN-AN",
    "Lakshadweep": "IN-LD",
    "Dadra & Nagar Haveli": "IN-DN",
}

TIME_PERIODS = {
    "Last 7 Days":   "now 7-d",
    "Last 15 Days":  "now 14-d",
    "Last 30 Days":  "today 1-m",
    "Last 3 Months": "today 3-m",
    "Last 6 Months": "today 6-m",
    "Last 1 Year":   "today 12-m",
}

PERIOD_KEYS = {
    "Last 7 Days":   "7_days",
    "Last 15 Days":  "15_days",
    "Last 30 Days":  "30_days",
    "Last 3 Months": "3_months",
    "Last 6 Months": "6_months",
    "Last 1 Year":   "1_year",
}

# Coastal/tropical states naturally score higher
HIGH_BASE_STATES = {"Kerala", "West Bengal", "Odisha", "Assam", "Tamil Nadu",
                    "Andhra Pradesh", "Goa", "Karnataka", "Telangana", "Maharashtra"}

COLOR_SCALE = "YlOrRd"
ACCENT = "#e74c3c"


# ── Custom CSS ───────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
      /* Main header */
      .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      }
      .hero h1 { font-size: 2.6rem; margin: 0 0 8px; letter-spacing: -1px; }
      .hero p  { font-size: 1.05rem; opacity: 0.8; margin: 0; }

      /* KPI cards */
      .kpi-grid { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
      .kpi-card {
        flex: 1; min-width: 150px;
        background: white;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-top: 4px solid var(--accent, #e74c3c);
      }
      .kpi-card .label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
      .kpi-card .value { font-size: 2rem; font-weight: 800; color: #1a1a2e; line-height: 1.1; margin-top: 4px; }
      .kpi-card .sub   { font-size: 0.82rem; color: #e74c3c; margin-top: 4px; font-weight: 600; }

      /* Section headers */
      .section-title {
        font-size: 1.2rem; font-weight: 700; color: #1a1a2e;
        border-left: 4px solid #e74c3c; padding-left: 12px;
        margin: 28px 0 16px;
      }

      /* Sidebar */
      [data-testid="stSidebar"] { background: #1a1a2e; }
      [data-testid="stSidebar"] * { color: white !important; }
      [data-testid="stSidebar"] .stSelectbox label,
      [data-testid="stSidebar"] .stMultiSelect label { color: #ccc !important; }

      /* Footer */
      .footer {
        text-align: center; color: #aaa; font-size: 0.78rem;
        margin-top: 48px; padding-top: 20px;
        border-top: 1px solid #eee;
      }

      /* Metric overrides */
      [data-testid="stMetric"] { background: white; border-radius: 12px; padding: 16px !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Data Layer ───────────────────────────────────────────────────────────────
def generate_demo_data(period_label: str, seed_offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic synthetic data for demo mode."""
    key = PERIOD_KEYS[period_label]
    seed = abs(hash(period_label)) % 10000 + seed_offset
    rng = np.random.default_rng(seed)

    rows = []
    for state, geo in INDIAN_STATES.items():
        base = 35 if state in HIGH_BASE_STATES else 12
        score = int(np.clip(base + rng.integers(-8, 52), 0, 100))
        rows.append({"State": state, "Geo_Code": geo, "Score": score})

    region_df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

    # Time series
    n_map = {"7_days": 7, "15_days": 15, "30_days": 30,
              "3_months": 90, "6_months": 180, "1_year": 52}
    freq_map = {"7_days": "D", "15_days": "D", "30_days": "D",
                "3_months": "D", "6_months": "D", "1_year": "W"}
    n = n_map[key]
    freq = freq_map[key]
    dates = pd.date_range(end=datetime.today(), periods=n, freq=freq)
    trend = 40 + 25 * np.sin(np.linspace(0, 2.5 * np.pi, len(dates))) + rng.normal(0, 7, len(dates))
    time_df = pd.DataFrame({"Date": dates, "Interest": np.clip(trend, 0, 100).astype(int)})

    return region_df, time_df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_data(period_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch real data from Google Trends via pytrends."""
    try:
        from pytrends.request import TrendReq
        tf = TIME_PERIODS[period_label]
        pt = TrendReq(hl="en-US", tz=330, timeout=(10, 30), retries=3, backoff_factor=0.8)

        # Region data
        pt.build_payload([KEYWORD], timeframe=tf, geo="IN")
        region_raw = pt.interest_by_region(resolution="REGION", inc_low_vol=True, inc_geo_code=True)
        region_raw = region_raw.reset_index()

        rows = []
        for state, geo in INDIAN_STATES.items():
            score = 0
            if "geoCode" in region_raw.columns:
                m = region_raw[region_raw["geoCode"] == geo]
                if not m.empty and KEYWORD in m.columns:
                    score = int(m[KEYWORD].values[0])
            elif "geoName" in region_raw.columns:
                m = region_raw[region_raw["geoName"] == state]
                if not m.empty and KEYWORD in m.columns:
                    score = int(m[KEYWORD].values[0])
            rows.append({"State": state, "Geo_Code": geo, "Score": score})

        region_df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

        # Time series
        time.sleep(1.5)
        pt.build_payload([KEYWORD], timeframe=tf, geo="IN")
        ts_raw = pt.interest_over_time()
        if not ts_raw.empty and KEYWORD in ts_raw.columns:
            time_df = ts_raw[[KEYWORD]].reset_index().rename(columns={"date": "Date", KEYWORD: "Interest"})
        else:
            time_df = pd.DataFrame(columns=["Date", "Interest"])

        return region_df, time_df

    except Exception as e:
        st.warning(f"⚠️ Google Trends API error: {e}. Falling back to demo data.")
        return generate_demo_data(period_label)


def get_data(period_label: str, demo_mode: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if demo_mode:
        return generate_demo_data(period_label)
    return fetch_live_data(period_label)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_periods_summary(demo_mode: bool) -> pd.DataFrame:
    """Build summary table across all periods."""
    rows = {state: {"State": state, "Geo_Code": geo} for state, geo in INDIAN_STATES.items()}

    for label in TIME_PERIODS.keys():
        region_df, _ = generate_demo_data(label) if demo_mode else fetch_live_data(label)
        for _, r in region_df.iterrows():
            rows[r["State"]][PERIOD_KEYS[label]] = r["Score"]

    return pd.DataFrame(list(rows.values()))


# ── Chart Builders ───────────────────────────────────────────────────────────
def chart_bar_top_states(region_df: pd.DataFrame, period_label: str, n: int = 15) -> go.Figure:
    df = region_df.head(n).sort_values("Score")
    fig = px.bar(
        df, x="Score", y="State", orientation="h",
        color="Score", color_continuous_scale=COLOR_SCALE,
        text="Score",
        title=f"🏆 Top {n} States by Mosquito Search Interest — {period_label}",
        labels={"Score": "Interest Score (0–100)", "State": ""},
    )
    fig.update_traces(textposition="outside", textfont_size=12)
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="white", paper_bgcolor="white",
        title_font_size=16, title_font_color="#1a1a2e",
        xaxis=dict(range=[0, 115], gridcolor="#f0f0f0"),
        yaxis=dict(tickfont=dict(size=12)),
        height=480, margin=dict(l=10, r=30, t=50, b=10),
    )
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.4,
                  annotation_text="Midpoint", annotation_position="top right")
    return fig


def chart_time_series(time_df: pd.DataFrame, period_label: str) -> go.Figure:
    if time_df.empty:
        return go.Figure()
    avg = time_df["Interest"].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_df["Date"], y=time_df["Interest"],
        mode="lines+markers",
        name="Interest",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.1)",
    ))
    fig.add_hline(
        y=avg, line_dash="dot", line_color="#3498db", line_width=1.5,
        annotation_text=f"Avg: {avg:.1f}", annotation_position="bottom right",
        annotation_font_color="#3498db",
    )
    fig.update_layout(
        title=f"📈 National Interest Over Time — {period_label}",
        title_font_size=16, title_font_color="#1a1a2e",
        xaxis_title="Date", yaxis_title="Interest Score",
        yaxis=dict(range=[0, 110], gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=360, margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    return fig


def chart_heatmap(summary_df: pd.DataFrame) -> go.Figure:
    period_cols = list(PERIOD_KEYS.values())
    cols_present = [c for c in period_cols if c in summary_df.columns]
    labels = {v: k for k, v in PERIOD_KEYS.items()}

    df_heat = summary_df.set_index("State")[cols_present].fillna(0)
    df_heat = df_heat.loc[df_heat.mean(axis=1).sort_values(ascending=False).index]

    fig = go.Figure(go.Heatmap(
        z=df_heat.values,
        x=[labels.get(c, c) for c in cols_present],
        y=df_heat.index.tolist(),
        colorscale=COLOR_SCALE,
        text=df_heat.values.astype(int),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z}<extra></extra>",
        colorbar=dict(title="Score"),
    ))
    fig.update_layout(
        title="🗺️ Heatmap — All States × All Time Periods",
        title_font_size=16, title_font_color="#1a1a2e",
        xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(600, len(df_heat) * 18 + 120),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white",
    )
    return fig


def chart_choropleth(region_df: pd.DataFrame, period_label: str) -> go.Figure:
    """India state choropleth using built-in Plotly geo data."""
    # Map state names to ISO codes for Plotly
    iso_map = {
        "Andhra Pradesh": "IN-AP", "Arunachal Pradesh": "IN-AR", "Assam": "IN-AS",
        "Bihar": "IN-BR", "Chhattisgarh": "IN-CT", "Goa": "IN-GA", "Gujarat": "IN-GJ",
        "Haryana": "IN-HR", "Himachal Pradesh": "IN-HP", "Jharkhand": "IN-JH",
        "Karnataka": "IN-KA", "Kerala": "IN-KL", "Madhya Pradesh": "IN-MP",
        "Maharashtra": "IN-MH", "Manipur": "IN-MN", "Meghalaya": "IN-ML",
        "Mizoram": "IN-MZ", "Nagaland": "IN-NL", "Odisha": "IN-OR",
        "Punjab": "IN-PB", "Rajasthan": "IN-RJ", "Sikkim": "IN-SK",
        "Tamil Nadu": "IN-TN", "Telangana": "IN-TG", "Tripura": "IN-TR",
        "Uttar Pradesh": "IN-UP", "Uttarakhand": "IN-UT", "West Bengal": "IN-WB",
        "Delhi": "IN-DL", "Jammu & Kashmir": "IN-JK", "Ladakh": "IN-LA",
        "Chandigarh": "IN-CH", "Puducherry": "IN-PY",
        "Andaman & Nicobar": "IN-AN", "Lakshadweep": "IN-LD",
        "Dadra & Nagar Haveli": "IN-DN",
    }
    df = region_df.copy()
    df["ISO"] = df["State"].map(iso_map)

    fig = px.choropleth(
        df, locations="ISO",
        locationmode="geojson-id",
        color="Score",
        hover_name="State",
        hover_data={"Score": True, "ISO": False},
        color_continuous_scale=COLOR_SCALE,
        range_color=[0, 100],
        scope="asia",
        title=f"🗺️ India State Map — {period_label}",
        labels={"Score": "Interest"},
    )
    fig.update_geos(
        center=dict(lat=22, lon=82),
        projection_scale=4.5,
        visible=False,
        showland=True, landcolor="#f5f5f0",
        showocean=True, oceancolor="#cce5ff",
        showlakes=True, lakecolor="#cce5ff",
        showcountries=True, countrycolor="#999",
        showsubunits=True, subunitcolor="#bbb",
        subunitwidth=1,
    )
    fig.update_layout(
        title_font_size=16, title_font_color="#1a1a2e",
        height=500, margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
        coloraxis_colorbar=dict(title="Score", thickness=14, len=0.7),
    )
    return fig


def chart_radar(summary_df: pd.DataFrame) -> go.Figure:
    period_cols = list(PERIOD_KEYS.values())
    cols_present = [c for c in period_cols if c in summary_df.columns]
    labels_map = {v: k for k, v in PERIOD_KEYS.items()}

    df = summary_df.set_index("State")[cols_present].fillna(0)
    top8 = df.loc[df.mean(axis=1).sort_values(ascending=False).head(8).index]

    colors = px.colors.qualitative.Plotly
    categories = [labels_map.get(c, c) for c in cols_present]

    fig = go.Figure()
    for i, (state, row) in enumerate(top8.iterrows()):
        vals = row.tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=state,
            line=dict(color=colors[i % len(colors)], width=2),
            fill="toself",
            fillcolor=colors[i % len(colors)],
            opacity=0.12,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#eee", tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor="white",
        ),
        title="🕸️ Radar — Top 8 States Across All Periods",
        title_font_size=16, title_font_color="#1a1a2e",
        showlegend=True,
        legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=-0.3),
        height=520, paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=80),
    )
    return fig


def chart_scatter_bubble(summary_df: pd.DataFrame) -> go.Figure:
    """Bubble chart: short-term vs long-term interest."""
    df = summary_df.copy()
    if "7_days" not in df.columns or "1_year" not in df.columns:
        return go.Figure()
    df = df.dropna(subset=["7_days", "1_year"])
    df["avg"] = df[[c for c in list(PERIOD_KEYS.values()) if c in df.columns]].mean(axis=1)
    df["Trending"] = df["7_days"] > df["1_year"]

    fig = px.scatter(
        df, x="1_year", y="7_days",
        size="avg", size_max=40,
        color="Trending",
        color_discrete_map={True: ACCENT, False: "#3498db"},
        hover_name="State",
        text="State",
        labels={"1_year": "1-Year Avg Interest", "7_days": "7-Day Interest", "Trending": "Recently Trending ↑"},
        title="🔍 Short-term vs Long-term Interest Bubble Chart",
    )
    fig.update_traces(textposition="top center", textfont=dict(size=9))
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                  line=dict(dash="dash", color="gray", width=1))
    fig.add_annotation(text="Above = Recently spiking", x=80, y=90,
                       showarrow=False, font=dict(color=ACCENT, size=10))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[0, 110], gridcolor="#f0f0f0"),
        yaxis=dict(range=[0, 110], gridcolor="#f0f0f0"),
        height=480, title_font_size=16, title_font_color="#1a1a2e",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def chart_box_distribution(summary_df: pd.DataFrame) -> go.Figure:
    """Box plot showing distribution of scores per period."""
    period_cols = list(PERIOD_KEYS.values())
    labels_map = {v: k for k, v in PERIOD_KEYS.items()}
    cols_present = [c for c in period_cols if c in summary_df.columns]

    fig = go.Figure()
    colors = px.colors.sequential.YlOrRd
    for i, col in enumerate(cols_present):
        color_idx = int(i / len(cols_present) * (len(colors) - 1))
        fig.add_trace(go.Box(
            y=summary_df[col].fillna(0),
            name=labels_map.get(col, col),
            marker_color=colors[color_idx],
            boxpoints="outliers",
            line_width=2,
        ))
    fig.update_layout(
        title="📦 Score Distribution by Time Period",
        title_font_size=16, title_font_color="#1a1a2e",
        yaxis=dict(title="Interest Score", range=[0, 110], gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[str, bool, int]:
    with st.sidebar:
        st.markdown("## 🦟 Mosquito Trends")
        st.markdown("**India State Analysis**")
        st.markdown("---")

        demo_mode = st.toggle("🧪 Demo Mode (no API)", value=True,
                              help="Use synthetic data. Disable for live Google Trends data.")
        if not demo_mode:
            st.info("📡 Live mode fetches from Google Trends. May be rate-limited.")

        st.markdown("---")
        period = st.selectbox(
            "⏱️ Time Period",
            options=list(TIME_PERIODS.keys()),
            index=4,
        )

        top_n = st.slider("🏆 Top N States (bar chart)", min_value=5, max_value=36, value=15, step=1)

        st.markdown("---")
        st.markdown("**About**")
        st.caption("Tracks Google search interest for 'Mosquito' across all 36 Indian States & UTs.")
        st.caption("Scores are relative (0–100). 100 = peak interest in region.")
        st.markdown("---")
        st.caption(f"🕐 Last updated: {datetime.now().strftime('%d %b %Y, %H:%M IST')}")

    return period, demo_mode, top_n


# ── KPI Cards ────────────────────────────────────────────────────────────────
def render_kpis(region_df: pd.DataFrame, time_df: pd.DataFrame):
    top_state = region_df.iloc[0]["State"]
    top_score = int(region_df.iloc[0]["Score"])
    avg_score = round(region_df["Score"].mean(), 1)
    active_states = int((region_df["Score"] > 0).sum())
    trend_dir = ""
    if not time_df.empty and len(time_df) >= 4:
        recent = time_df["Interest"].iloc[-3:].mean()
        earlier = time_df["Interest"].iloc[-6:-3].mean()
        trend_dir = "↑ Rising" if recent > earlier else "↓ Falling"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏅 Top State", top_state, f"Score: {top_score}")
    with c2:
        st.metric("📊 Avg Score (All States)", avg_score)
    with c3:
        st.metric("🗺️ States with Interest", f"{active_states}/36")
    with c4:
        st.metric("📈 Recent Trend", trend_dir if trend_dir else "—")


# ── Main App ─────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # Hero banner
    st.markdown("""
    <div class="hero">
      <h1>🦟 Mosquito Google Trends</h1>
      <p>Search interest analysis across Indian States & Union Territories • Powered by Google Trends</p>
    </div>
    """, unsafe_allow_html=True)

    period, demo_mode, top_n = render_sidebar()

    if demo_mode:
        st.info("🧪 **Demo Mode** — Showing synthetic data. Toggle 'Demo Mode' off in the sidebar for live Google Trends data.", icon="ℹ️")

    # ── Fetch single-period data ─────────────────────────────────────────────
    with st.spinner(f"Loading data for {period}…"):
        region_df, time_df = get_data(period, demo_mode)

    # ── KPI row ──────────────────────────────────────────────────────────────
    render_kpis(region_df, time_df)
    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Period Analysis", "🗺️ Cross-Period View", "🔍 Deep Dive", "📋 Data Table"])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 – Single period charts
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        col_l, col_r = st.columns([1.1, 1], gap="large")

        with col_l:
            st.markdown(f'<div class="section-title">Top {top_n} States</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_bar_top_states(region_df, period, top_n),
                            use_container_width=True, config={"displayModeBar": False})

        with col_r:
            st.markdown('<div class="section-title">India Map</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_choropleth(region_df, period),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">National Interest Over Time</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_time_series(time_df, period),
                        use_container_width=True, config={"displayModeBar": False})

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 – Cross-period heatmap, radar, box
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        with st.spinner("Loading all-period summary…"):
            summary_df = fetch_all_periods_summary(demo_mode)

        st.plotly_chart(chart_heatmap(summary_df), use_container_width=True)

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.plotly_chart(chart_radar(summary_df), use_container_width=True)
        with col_b:
            st.plotly_chart(chart_box_distribution(summary_df), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 – Deep dive: bubble + multi-period comparison
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
        with st.spinner("Building deep-dive charts…"):
            if "summary_df" not in dir():
                summary_df = fetch_all_periods_summary(demo_mode)

        st.plotly_chart(chart_scatter_bubble(summary_df), use_container_width=True)

        st.markdown('<div class="section-title">Compare States Across Periods</div>', unsafe_allow_html=True)
        selected_states = st.multiselect(
            "Select states to compare:",
            options=list(INDIAN_STATES.keys()),
            default=["Kerala", "West Bengal", "Delhi", "Maharashtra", "Tamil Nadu"],
        )

        if selected_states:
            period_cols = [c for c in list(PERIOD_KEYS.values()) if c in summary_df.columns]
            labels_map = {v: k for k, v in PERIOD_KEYS.items()}
            df_sel = summary_df[summary_df["State"].isin(selected_states)].set_index("State")[period_cols]
            df_melt = df_sel.reset_index().melt(id_vars="State", var_name="Period", value_name="Score")
            df_melt["Period"] = df_melt["Period"].map(labels_map)

            fig_line = px.line(
                df_melt, x="Period", y="Score", color="State",
                markers=True, line_shape="spline",
                title="Selected States — Score Across All Time Periods",
                labels={"Score": "Interest Score"},
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig_line.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(range=[0, 110], gridcolor="#f0f0f0"),
                height=400, title_font_size=15, title_font_color="#1a1a2e",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_line, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 – Data table + download
    # ═══════════════════════════════════════════════════════════════════════
    with tab4:
        with st.spinner("Preparing data table…"):
            if "summary_df" not in dir():
                summary_df = fetch_all_periods_summary(demo_mode)

        period_cols = [c for c in list(PERIOD_KEYS.values()) if c in summary_df.columns]
        labels_map = {v: k for k, v in PERIOD_KEYS.items()}
        display_df = summary_df[["State"] + period_cols].copy()
        display_df = display_df.rename(columns=labels_map)
        display_df = display_df.sort_values("Last 1 Year", ascending=False).reset_index(drop=True)
        display_df.index += 1
        display_df.index.name = "Rank"

        # Color map for score cells
        def color_score(val):
            if val >= 75: return "background-color:#FF6B6B;color:white;font-weight:bold"
            if val >= 50: return "background-color:#FFC300;color:#333;font-weight:bold"
            if val >= 25: return "background-color:#82e0aa;color:#333"
            if val > 0:   return "background-color:#aed6f1;color:#333"
            return "color:#bbb"

        num_cols = [c for c in display_df.columns if c != "State"]
        st.dataframe(
            display_df.style.applymap(color_score, subset=num_cols),
            use_container_width=True,
            height=600,
        )

        st.markdown("**Legend:** 🔴 75–100 Very High &nbsp;|&nbsp; 🟡 50–74 High &nbsp;|&nbsp; 🟢 25–49 Moderate &nbsp;|&nbsp; 🔵 1–24 Low &nbsp;|&nbsp; ⬜ 0 No Data")
        st.markdown("---")

        csv = display_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"mosquito_trends_india_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
      🦟 Mosquito Trends India &nbsp;|&nbsp; Data: Google Trends via pytrends &nbsp;|&nbsp;
      Scores are relative (0=lowest, 100=peak interest in region) &nbsp;|&nbsp;
      <a href="https://github.com" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
