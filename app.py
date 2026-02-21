"""
🦟 Mosquito Google Trends – India State Analysis
Streamlit App | Deploy via GitHub → Streamlit Cloud

All fixes included:
  • urllib3 method_whitelist → allowed_methods compatibility patch
  • No fragile internal-attribute swapping
  • Daily trend line with momentum banner
  • Demo mode + live Google Trends mode
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Page config — must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="🦟 Mosquito Trends India",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
KEYWORD = "Mosquito"

INDIAN_STATES = {
    "Andhra Pradesh":       "IN-AP",
    "Arunachal Pradesh":    "IN-AR",
    "Assam":                "IN-AS",
    "Bihar":                "IN-BR",
    "Chhattisgarh":         "IN-CT",
    "Goa":                  "IN-GA",
    "Gujarat":              "IN-GJ",
    "Haryana":              "IN-HR",
    "Himachal Pradesh":     "IN-HP",
    "Jharkhand":            "IN-JH",
    "Karnataka":            "IN-KA",
    "Kerala":               "IN-KL",
    "Madhya Pradesh":       "IN-MP",
    "Maharashtra":          "IN-MH",
    "Manipur":              "IN-MN",
    "Meghalaya":            "IN-ML",
    "Mizoram":              "IN-MZ",
    "Nagaland":             "IN-NL",
    "Odisha":               "IN-OR",
    "Punjab":               "IN-PB",
    "Rajasthan":            "IN-RJ",
    "Sikkim":               "IN-SK",
    "Tamil Nadu":           "IN-TN",
    "Telangana":            "IN-TG",
    "Tripura":              "IN-TR",
    "Uttar Pradesh":        "IN-UP",
    "Uttarakhand":          "IN-UT",
    "West Bengal":          "IN-WB",
    "Delhi":                "IN-DL",
    "Jammu & Kashmir":      "IN-JK",
    "Ladakh":               "IN-LA",
    "Chandigarh":           "IN-CH",
    "Puducherry":           "IN-PY",
    "Andaman & Nicobar":    "IN-AN",
    "Lakshadweep":          "IN-LD",
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

HIGH_BASE_STATES = {
    "Kerala", "West Bengal", "Odisha", "Assam", "Tamil Nadu",
    "Andhra Pradesh", "Goa", "Karnataka", "Telangana", "Maharashtra",
}

COLOR_SCALE = "YlOrRd"
ACCENT      = "#e74c3c"


# ─────────────────────────────────────────────────────────────────────────────
# urllib3 COMPATIBILITY PATCH
# pytrends calls urllib3.Retry(method_whitelist=...) which was renamed to
# allowed_methods in urllib3 >= 1.26. Monkey-patch once at module load.
# ─────────────────────────────────────────────────────────────────────────────
def _patch_urllib3_retry() -> None:
    try:
        import inspect
        from urllib3.util.retry import Retry

        _orig = Retry.__init__

        def _patched(self, *args, **kwargs):
            if "method_whitelist" in kwargs:
                kwargs.setdefault("allowed_methods", kwargs.pop("method_whitelist"))
            _orig(self, *args, **kwargs)

        if "method_whitelist" not in inspect.signature(_orig).parameters:
            Retry.__init__ = _patched
    except Exception:
        pass


_patch_urllib3_retry()


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown("""
    <style>
      .hero {
        background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        border-radius:16px; padding:36px 40px; margin-bottom:28px;
        color:white; text-align:center;
        box-shadow:0 8px 32px rgba(0,0,0,.3);
      }
      .hero h1 { font-size:2.6rem; margin:0 0 8px; letter-spacing:-1px; }
      .hero p  { font-size:1.05rem; opacity:.8; margin:0; }

      .section-title {
        font-size:1.15rem; font-weight:700; color:#1a1a2e;
        border-left:4px solid #e74c3c; padding-left:12px;
        margin:24px 0 12px;
      }

      [data-testid="stSidebar"] { background:#1a1a2e !important; }
      [data-testid="stSidebar"] * { color:white !important; }

      [data-testid="stMetric"] {
        background:white; border-radius:12px;
        padding:16px !important;
        box-shadow:0 2px 8px rgba(0,0,0,.06);
      }

      .footer {
        text-align:center; color:#aaa; font-size:.78rem;
        margin-top:48px; padding-top:20px; border-top:1px solid #eee;
      }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
def _make_time_series(key: str, rng) -> pd.DataFrame:
    n_map    = {"7_days":7,"15_days":15,"30_days":30,
                "3_months":90,"6_months":180,"1_year":52}
    freq_map = {"7_days":"D","15_days":"D","30_days":"D",
                "3_months":"D","6_months":"D","1_year":"W"}
    dates  = pd.date_range(end=datetime.today(),
                           periods=n_map[key], freq=freq_map[key])
    n      = len(dates)
    trend  = (40 + 25 * np.sin(np.linspace(0, 2.5 * np.pi, n))
              + rng.normal(0, 7, n))
    return pd.DataFrame({"Date": dates,
                         "Interest": np.clip(trend, 0, 100).astype(int)})


def generate_demo_data(period_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    key  = PERIOD_KEYS[period_label]
    rng  = np.random.default_rng(abs(hash(period_label)) % 10_000)
    rows = []
    for state, geo in INDIAN_STATES.items():
        base  = 35 if state in HIGH_BASE_STATES else 12
        score = int(np.clip(base + rng.integers(-8, 52), 0, 100))
        rows.append({"State": state, "Geo_Code": geo, "Score": score})
    region_df = (pd.DataFrame(rows)
                 .sort_values("Score", ascending=False)
                 .reset_index(drop=True))
    time_df   = _make_time_series(key, rng)
    return region_df, time_df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_data(period_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from pytrends.request import TrendReq
        tf = TIME_PERIODS[period_label]
        pt = TrendReq(hl="en-US", tz=330, timeout=(10, 30),
                      retries=2, backoff_factor=0.5)

        pt.build_payload([KEYWORD], timeframe=tf, geo="IN")
        raw = pt.interest_by_region(
            resolution="REGION", inc_low_vol=True, inc_geo_code=True
        ).reset_index()

        rows = []
        for state, geo in INDIAN_STATES.items():
            score = 0
            if "geoCode" in raw.columns:
                m = raw[raw["geoCode"] == geo]
            elif "geoName" in raw.columns:
                m = raw[raw["geoName"] == state]
            else:
                m = pd.DataFrame()
            if not m.empty and KEYWORD in m.columns:
                score = int(m[KEYWORD].values[0])
            rows.append({"State": state, "Geo_Code": geo, "Score": score})

        region_df = (pd.DataFrame(rows)
                     .sort_values("Score", ascending=False)
                     .reset_index(drop=True))

        time.sleep(1.5)
        pt.build_payload([KEYWORD], timeframe=tf, geo="IN")
        ts = pt.interest_over_time()
        if not ts.empty and KEYWORD in ts.columns:
            time_df = (ts[[KEYWORD]].reset_index()
                       .rename(columns={"date": "Date", KEYWORD: "Interest"}))
        else:
            time_df = pd.DataFrame(columns=["Date", "Interest"])

        return region_df, time_df

    except Exception as exc:
        st.warning(f"⚠️ Google Trends error: {exc} — showing demo data.", icon="⚠️")
        return generate_demo_data(period_label)


def get_data(period_label: str, demo_mode: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (generate_demo_data(period_label) if demo_mode
            else fetch_live_data(period_label))


@st.cache_data(ttl=3600, show_spinner=False)
def all_periods_summary(demo_mode: bool) -> pd.DataFrame:
    records = {s: {"State": s, "Geo_Code": g} for s, g in INDIAN_STATES.items()}
    for label in TIME_PERIODS:
        col = PERIOD_KEYS[label]
        rdf, _ = (generate_demo_data(label) if demo_mode
                  else fetch_live_data(label))
        for _, row in rdf.iterrows():
            records[row["State"]][col] = row["Score"]
    return pd.DataFrame(list(records.values()))


# ─────────────────────────────────────────────────────────────────────────────
# TREND MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────
def analyse_momentum(time_df: pd.DataFrame) -> dict:
    """Compare recent third vs earlier portion; return status dict."""
    if time_df.empty or len(time_df) < 4:
        return dict(status="neutral", label="Not enough data",
                    color="#95a5a6", emoji="➖", pct_change=0.0, slope=0.0)

    vals    = time_df["Interest"].values.astype(float)
    n       = len(vals)
    split   = max(2, n // 3)
    recent  = vals[-split:].mean()
    earlier = vals[:-split].mean()
    pct     = (recent - earlier) / (earlier + 1e-9) * 100

    window  = min(7, n)
    slope   = float(np.polyfit(np.arange(window), vals[-window:], 1)[0])

    if   pct >= 20 and slope >  0.5:
        return dict(status="surging",  label="Surging",
                    color="#c0392b", emoji="🚀", pct_change=pct, slope=slope)
    elif pct >=  8 and slope >= 0:
        return dict(status="rising",   label="Picking Up",
                    color="#d35400", emoji="📈", pct_change=pct, slope=slope)
    elif pct <= -20 and slope < -0.5:
        return dict(status="crashing", label="Dropping Fast",
                    color="#1a5276", emoji="📉", pct_change=pct, slope=slope)
    elif pct <=  -8 and slope <= 0:
        return dict(status="falling",  label="Cooling Down",
                    color="#2471a3", emoji="🔽", pct_change=pct, slope=slope)
    else:
        return dict(status="stable",   label="Stable",
                    color="#1e8449", emoji="➡️", pct_change=pct, slope=slope)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def chart_top_states(region_df: pd.DataFrame, period_label: str, n: int) -> go.Figure:
    df  = region_df.head(n).sort_values("Score")
    fig = px.bar(
        df, x="Score", y="State", orientation="h",
        color="Score", color_continuous_scale=COLOR_SCALE, text="Score",
        title=f"🏆 Top {n} States — {period_label}",
        labels={"Score": "Interest (0–100)", "State": ""},
    )
    fig.update_traces(textposition="outside", textfont_size=12)
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="white", paper_bgcolor="white",
        title_font=dict(size=16, color="#1a1a2e"),
        xaxis=dict(range=[0, 115], gridcolor="#f0f0f0"),
        yaxis=dict(tickfont=dict(size=12)),
        height=max(380, n * 28 + 80),
        margin=dict(l=10, r=40, t=50, b=10),
    )
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.3,
                  annotation_text="Mid", annotation_position="top right")
    return fig


def chart_daily_trend(time_df: pd.DataFrame, period_label: str) -> go.Figure:
    """
    Segment-coloured daily trend line:
      Red  = above period average  |  Blue = below period average
      Gold dotted = rolling average  |  ★ peak  |  ▼ trough
      Momentum badge in top-right corner  |  Range slider
    """
    if time_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No time-series data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="#aaa"))
        return fig

    df   = (time_df.copy()
            .assign(Date=lambda d: pd.to_datetime(d["Date"]))
            .sort_values("Date")
            .reset_index(drop=True))
    vals = df["Interest"].values.astype(float)
    avg  = float(vals.mean())
    n    = len(df)

    win        = min(7, max(3, n // 5))
    df["Roll"] = df["Interest"].rolling(window=win, center=True, min_periods=1).mean()

    peak_i   = int(np.argmax(vals))
    trough_i = int(np.argmin(vals))

    fig = go.Figure()

    # Background shading
    fig.add_hrect(y0=avg, y1=106, fillcolor="rgba(231,76,60,0.05)",
                  layer="below", line_width=0)
    fig.add_hrect(y0=0,   y1=avg, fillcolor="rgba(52,152,219,0.05)",
                  layer="below", line_width=0)

    # Coloured area fills per segment
    for i in range(n - 1):
        mid  = (vals[i] + vals[i + 1]) / 2
        fill = ("rgba(231,76,60,0.13)" if mid >= avg
                else "rgba(52,152,219,0.13)")
        fig.add_trace(go.Scatter(
            x=[df["Date"].iloc[i], df["Date"].iloc[i+1],
               df["Date"].iloc[i+1], df["Date"].iloc[i]],
            y=[vals[i], vals[i+1], 0, 0],
            fill="toself", fillcolor=fill,
            mode="none", showlegend=False, hoverinfo="skip",
        ))

    # Coloured line segments
    for i in range(n - 1):
        mid = (vals[i] + vals[i + 1]) / 2
        col = "#e74c3c" if mid >= avg else "#3498db"
        fig.add_trace(go.Scatter(
            x=[df["Date"].iloc[i], df["Date"].iloc[i+1]],
            y=[vals[i], vals[i+1]],
            mode="lines", line=dict(color=col, width=2.8),
            showlegend=False, hoverinfo="skip",
        ))

    # Hover-enabled dot trace (invisible dots drive the unified tooltip)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Interest"],
        mode="markers",
        marker=dict(
            size=5,
            color=["#e74c3c" if v >= avg else "#3498db" for v in vals],
            line=dict(width=1, color="white"),
        ),
        name="Daily Interest",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Interest: <b>%{y}</b><extra></extra>",
    ))

    # Rolling average ribbon
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Roll"],
        mode="lines", name=f"{win}-day rolling avg",
        line=dict(color="#f39c12", width=2, dash="dot"),
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            f"{win}-day avg: <b>%{{y:.1f}}</b><extra></extra>"
        ),
    ))

    # Period average reference line
    fig.add_hline(
        y=avg, line_dash="dash", line_color="#aab7b8", line_width=1.2,
        annotation_text=f"Period avg {avg:.0f}",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#888"),
    )

    # Peak marker
    fig.add_trace(go.Scatter(
        x=[df["Date"].iloc[peak_i]], y=[vals[peak_i]],
        mode="markers+text",
        marker=dict(symbol="star", size=15, color="#e74c3c",
                    line=dict(width=1.5, color="white")),
        text=[f"  Peak {int(vals[peak_i])}"],
        textposition="top right", textfont=dict(size=10, color="#e74c3c"),
        name="Peak",
        hovertemplate=f"Peak: {int(vals[peak_i])}<extra></extra>",
    ))

    # Trough marker
    if trough_i != peak_i:
        fig.add_trace(go.Scatter(
            x=[df["Date"].iloc[trough_i]], y=[vals[trough_i]],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=13, color="#3498db",
                        line=dict(width=1.5, color="white")),
            text=[f"  Low {int(vals[trough_i])}"],
            textposition="bottom right", textfont=dict(size=10, color="#3498db"),
            name="Low",
            hovertemplate=f"Low: {int(vals[trough_i])}<extra></extra>",
        ))

    # Momentum badge in chart corner
    mom     = analyse_momentum(df)
    pct_lbl = (f"+{mom['pct_change']:.1f}%" if mom["pct_change"] >= 0
               else f"{mom['pct_change']:.1f}%")
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.97,
        text=(f"<b>{mom['emoji']}  {mom['label']}</b><br>"
              f"<span style='font-size:11px'>Recent vs earlier: {pct_lbl}</span>"),
        showarrow=False,
        bgcolor=mom["color"], bordercolor=mom["color"],
        borderwidth=2, borderpad=8,
        font=dict(color="white", size=13),
        align="right", xanchor="right", yanchor="top", opacity=0.92,
    )

    fig.update_layout(
        title=dict(text=f"📅 Daily Trend — {period_label}",
                   font=dict(size=17, color="#1a1a2e")),
        xaxis=dict(
            title="Date", showgrid=True, gridcolor="#f0f0f0",
            tickformat="%d %b",
            rangeslider=dict(visible=True, thickness=0.06),
        ),
        yaxis=dict(
            title="Search Interest (0–100)",
            range=[0, 112], gridcolor="#f0f0f0", zeroline=False,
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        height=490,
        margin=dict(l=10, r=10, t=55, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    return fig


def chart_choropleth(region_df: pd.DataFrame, period_label: str) -> go.Figure:
    fig = px.choropleth(
        region_df, locations="Geo_Code", locationmode="geojson-id",
        color="Score", hover_name="State",
        hover_data={"Score": True, "Geo_Code": False},
        color_continuous_scale=COLOR_SCALE, range_color=[0, 100],
        scope="asia", title=f"🗺️ India Map — {period_label}",
        labels={"Score": "Interest"},
    )
    fig.update_geos(
        center=dict(lat=22, lon=82), projection_scale=4.5, visible=False,
        showland=True, landcolor="#f5f5f0",
        showocean=True, oceancolor="#cce5ff",
        showlakes=True, lakecolor="#cce5ff",
        showcountries=True, countrycolor="#999",
        showsubunits=True, subunitcolor="#bbb", subunitwidth=1,
    )
    fig.update_layout(
        title_font=dict(size=16, color="#1a1a2e"),
        height=500, margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
        coloraxis_colorbar=dict(title="Score", thickness=14, len=0.7),
    )
    return fig


def chart_heatmap(summary_df: pd.DataFrame) -> go.Figure:
    pcols  = [c for c in PERIOD_KEYS.values() if c in summary_df.columns]
    rev    = {v: k for k, v in PERIOD_KEYS.items()}
    heat   = (summary_df.set_index("State")[pcols].fillna(0).astype(int))
    heat   = heat.loc[heat.mean(axis=1).sort_values(ascending=False).index]
    fig    = go.Figure(go.Heatmap(
        z=heat.values,
        x=[rev[c] for c in pcols],
        y=heat.index.tolist(),
        colorscale=COLOR_SCALE,
        text=heat.values, texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z}<extra></extra>",
        colorbar=dict(title="Score"),
    ))
    fig.update_layout(
        title=dict(text="🗺️ Heatmap — All States × All Time Periods",
                   font=dict(size=16, color="#1a1a2e")),
        xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(600, len(heat) * 18 + 120),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white",
    )
    return fig


def chart_radar(summary_df: pd.DataFrame) -> go.Figure:
    pcols  = [c for c in PERIOD_KEYS.values() if c in summary_df.columns]
    rev    = {v: k for k, v in PERIOD_KEYS.items()}
    df     = summary_df.set_index("State")[pcols].fillna(0)
    top8   = df.loc[df.mean(axis=1).sort_values(ascending=False).head(8).index]
    cats   = [rev[c] for c in pcols]
    colors = px.colors.qualitative.Plotly
    fig    = go.Figure()
    for i, (state, row) in enumerate(top8.iterrows()):
        v = row.tolist()
        fig.add_trace(go.Scatterpolar(
            r=v + [v[0]], theta=cats + [cats[0]],
            name=state,
            line=dict(color=colors[i % len(colors)], width=2),
            fill="toself", fillcolor=colors[i % len(colors)], opacity=0.1,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="#eee", tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor="white",
        ),
        title=dict(text="🕸️ Radar — Top 8 States Across All Periods",
                   font=dict(size=16, color="#1a1a2e")),
        showlegend=True,
        legend=dict(font=dict(size=10), orientation="h",
                    yanchor="bottom", y=-0.32),
        height=520, paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=80),
    )
    return fig


def chart_box(summary_df: pd.DataFrame) -> go.Figure:
    pcols = [c for c in PERIOD_KEYS.values() if c in summary_df.columns]
    rev   = {v: k for k, v in PERIOD_KEYS.items()}
    reds  = px.colors.sequential.YlOrRd
    fig   = go.Figure()
    for i, col in enumerate(pcols):
        ci = int(i / len(pcols) * (len(reds) - 1))
        fig.add_trace(go.Box(
            y=summary_df[col].fillna(0), name=rev[col],
            marker_color=reds[ci], boxpoints="outliers", line_width=2,
        ))
    fig.update_layout(
        title=dict(text="📦 Score Distribution by Time Period",
                   font=dict(size=16, color="#1a1a2e")),
        yaxis=dict(title="Interest Score", range=[0, 110], gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


def chart_bubble(summary_df: pd.DataFrame) -> go.Figure:
    df = summary_df.copy()
    if not {"7_days", "1_year"}.issubset(df.columns):
        return go.Figure()
    pcols = [c for c in PERIOD_KEYS.values() if c in df.columns]
    df["avg"]      = df[pcols].mean(axis=1)
    df["Trending"] = df["7_days"] > df["1_year"]
    fig = px.scatter(
        df.dropna(subset=["7_days", "1_year"]),
        x="1_year", y="7_days", size="avg", size_max=40,
        color="Trending",
        color_discrete_map={True: ACCENT, False: "#3498db"},
        hover_name="State", text="State",
        title="🔍 Short-term vs Long-term Interest",
        labels={"1_year": "1-Year Interest", "7_days": "7-Day Interest",
                "Trending": "Spiking Recently"},
    )
    fig.update_traces(textposition="top center", textfont=dict(size=9))
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                  line=dict(dash="dash", color="gray", width=1))
    fig.add_annotation(text="↑ Recently spiking", x=75, y=93,
                       showarrow=False, font=dict(color=ACCENT, size=10))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[0, 110], gridcolor="#f0f0f0"),
        yaxis=dict(range=[0, 110], gridcolor="#f0f0f0"),
        height=480, title_font=dict(size=16, color="#1a1a2e"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[str, bool, int]:
    with st.sidebar:
        st.markdown("## 🦟 Mosquito Trends")
        st.markdown("**India State Analysis**")
        st.divider()
        demo_mode = st.toggle("🧪 Demo Mode (no API)", value=True,
                              help="Synthetic data. Disable for live Google Trends.")
        if not demo_mode:
            st.info("📡 Live mode — may be rate-limited.", icon="ℹ️")
        st.divider()
        period = st.selectbox("⏱️ Time Period", list(TIME_PERIODS.keys()), index=4)
        top_n  = st.slider("🏆 Top N States", min_value=5, max_value=36, value=15)
        st.divider()
        st.caption("Scores: 0 = lowest, 100 = peak interest in region.")
        st.caption(f"🕐 {datetime.now().strftime('%d %b %Y, %H:%M IST')}")
    return period, demo_mode, top_n


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
def render_kpis(region_df: pd.DataFrame, time_df: pd.DataFrame) -> None:
    mom = analyse_momentum(time_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏅 Top State",
              region_df.iloc[0]["State"],
              f"Score {int(region_df.iloc[0]['Score'])}")
    c2.metric("📊 Avg Score (All States)",
              round(float(region_df["Score"].mean()), 1))
    c3.metric("🗺️ States with Activity",
              f"{int((region_df['Score'] > 0).sum())} / 36")
    c4.metric("📈 Trend Momentum",
              f"{mom['emoji']} {mom['label']}")


# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM BANNER
# ─────────────────────────────────────────────────────────────────────────────
def render_momentum_banner(time_df: pd.DataFrame) -> None:
    mom       = analyse_momentum(time_df)
    pct_str   = (f"+{mom['pct_change']:.1f}%" if mom["pct_change"] >= 0
                 else f"{mom['pct_change']:.1f}%")
    slope_dir = "▲" if mom["slope"] > 0 else "▼"
    slope_str = f"Slope {slope_dir} {abs(mom['slope']):.2f} pts / day"
    bg = {
        "surging":  "linear-gradient(90deg,#c0392b,#e74c3c)",
        "rising":   "linear-gradient(90deg,#d35400,#e67e22)",
        "stable":   "linear-gradient(90deg,#1e8449,#27ae60)",
        "falling":  "linear-gradient(90deg,#1a5276,#2471a3)",
        "crashing": "linear-gradient(90deg,#154360,#1a5276)",
    }.get(mom["status"], "linear-gradient(90deg,#566573,#95a5a6)")

    st.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:18px 28px;
                display:flex;align-items:center;justify-content:space-between;
                margin:16px 0 8px;box-shadow:0 3px 10px rgba(0,0,0,.18)">
      <div style="display:flex;align-items:center;gap:14px">
        <span style="font-size:2.2rem;line-height:1">{mom['emoji']}</span>
        <div>
          <div style="color:white;font-size:1.45rem;font-weight:800;
                      line-height:1.1">{mom['label']}</div>
          <div style="color:rgba(255,255,255,.72);font-size:.82rem;
                      margin-top:2px">Trend Momentum</div>
        </div>
      </div>
      <div style="text-align:right">
        <div style="color:white;font-size:1.15rem;font-weight:700">{pct_str}</div>
        <div style="color:rgba(255,255,255,.72);font-size:.82rem">
          recent vs earlier period</div>
        <div style="color:rgba(255,255,255,.55);font-size:.78rem;
                    margin-top:2px">{slope_str}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()

    st.markdown("""
    <div class="hero">
      <h1>🦟 Mosquito Google Trends</h1>
      <p>Search interest across Indian States &amp; Union Territories
         &nbsp;·&nbsp; Powered by Google Trends</p>
    </div>
    """, unsafe_allow_html=True)

    period, demo_mode, top_n = render_sidebar()

    if demo_mode:
        st.info(
            "🧪 **Demo Mode** — Synthetic data shown. "
            "Disable **'Demo Mode'** in the sidebar for live Google Trends data.",
            icon="ℹ️",
        )

    with st.spinner(f"Loading {period} data…"):
        region_df, time_df = get_data(period, demo_mode)

    render_kpis(region_df, time_df)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Period Analysis",
        "🗺️ Cross-Period View",
        "🔍 Deep Dive",
        "📋 Data Table",
    ])

    # ── TAB 1: Period Analysis ────────────────────────────────────────────────
    with tab1:
        col_l, col_r = st.columns([1.1, 1], gap="large")

        with col_l:
            st.markdown(f'<div class="section-title">Top {top_n} States</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(chart_top_states(region_df, period, top_n),
                            use_container_width=True,
                            config={"displayModeBar": False})

        with col_r:
            st.markdown('<div class="section-title">India Map</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(chart_choropleth(region_df, period),
                            use_container_width=True,
                            config={"displayModeBar": False})

        # Momentum banner
        render_momentum_banner(time_df)

        # Daily trend line
        st.markdown('<div class="section-title">Daily Trend Line</div>',
                    unsafe_allow_html=True)
        st.caption(
            "🔴 Above avg  •  🔵 Below avg  •  "
            "🟡·· Rolling avg  •  ★ Peak  •  ▼ Low"
        )
        st.plotly_chart(chart_daily_trend(time_df, period),
                        use_container_width=True,
                        config={"displayModeBar": False})

    # ── TAB 2: Cross-Period View ──────────────────────────────────────────────
    with tab2:
        with st.spinner("Loading all-period summary…"):
            summary_df = all_periods_summary(demo_mode)

        st.plotly_chart(chart_heatmap(summary_df), use_container_width=True)

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.plotly_chart(chart_radar(summary_df), use_container_width=True)
        with col_b:
            st.plotly_chart(chart_box(summary_df), use_container_width=True)

    # ── TAB 3: Deep Dive ─────────────────────────────────────────────────────
    with tab3:
        if "summary_df" not in dir():
            with st.spinner("Building charts…"):
                summary_df = all_periods_summary(demo_mode)

        st.plotly_chart(chart_bubble(summary_df), use_container_width=True)

        st.markdown(
            '<div class="section-title">Compare States Across Periods</div>',
            unsafe_allow_html=True,
        )
        selected = st.multiselect(
            "Select states to compare:",
            options=list(INDIAN_STATES.keys()),
            default=["Kerala", "West Bengal", "Delhi", "Maharashtra", "Tamil Nadu"],
        )
        if selected:
            pcols   = [c for c in PERIOD_KEYS.values() if c in summary_df.columns]
            rev     = {v: k for k, v in PERIOD_KEYS.items()}
            df_sel  = (summary_df[summary_df["State"].isin(selected)]
                       .set_index("State")[pcols])
            df_melt = (df_sel.reset_index()
                       .melt(id_vars="State", var_name="Period", value_name="Score"))
            df_melt["Period"] = df_melt["Period"].map(rev)
            fig_cmp = px.line(
                df_melt, x="Period", y="Score", color="State",
                markers=True, line_shape="spline",
                title="Selected States — Score Across All Time Periods",
                labels={"Score": "Interest Score"},
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig_cmp.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(range=[0, 110], gridcolor="#f0f0f0"),
                height=420, title_font=dict(size=15, color="#1a1a2e"),
                legend=dict(orientation="h", y=-0.28),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

    # ── TAB 4: Data Table ────────────────────────────────────────────────────
    with tab4:
        if "summary_df" not in dir():
            with st.spinner("Preparing table…"):
                summary_df = all_periods_summary(demo_mode)

        pcols      = [c for c in PERIOD_KEYS.values() if c in summary_df.columns]
        rev        = {v: k for k, v in PERIOD_KEYS.items()}
        display_df = (summary_df[["State"] + pcols]
                      .copy()
                      .rename(columns=rev)
                      .sort_values("Last 1 Year", ascending=False)
                      .reset_index(drop=True))
        display_df.index      += 1
        display_df.index.name  = "Rank"

        def _cell_style(val):
            if   val >= 75: return "background-color:#FF6B6B;color:white;font-weight:bold"
            elif val >= 50: return "background-color:#FFC300;color:#333;font-weight:bold"
            elif val >= 25: return "background-color:#82e0aa;color:#333"
            elif val  >  0: return "background-color:#aed6f1;color:#333"
            return "color:#ccc"

        num_cols = [c for c in display_df.columns if c != "State"]
        st.dataframe(
            display_df.style.applymap(_cell_style, subset=num_cols),
            use_container_width=True, height=600,
        )
        st.markdown(
            "**Legend:** "
            "🔴 75–100 Very High &nbsp;|&nbsp; "
            "🟡 50–74 High &nbsp;|&nbsp; "
            "🟢 25–49 Moderate &nbsp;|&nbsp; "
            "🔵 1–24 Low &nbsp;|&nbsp; "
            "⬜ 0 No data"
        )
        st.divider()
        st.download_button(
            label="⬇️ Download CSV",
            data=display_df.to_csv(index=True).encode("utf-8"),
            file_name=f"mosquito_trends_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    st.markdown("""
    <div class="footer">
      🦟 Mosquito Trends India &nbsp;|&nbsp;
      Data via Google Trends / pytrends &nbsp;|&nbsp;
      Scores are relative — 0 = lowest, 100 = peak interest in region
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
