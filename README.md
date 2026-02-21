# 🦟 Mosquito Google Trends — India State Analysis

Interactive Streamlit dashboard tracking Google search interest for **"Mosquito"**
across all 36 Indian States & Union Territories for six time windows.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📊 Features

| Feature | Detail |
|---|---|
| **6 Time Periods** | 7 days · 15 days · 30 days · 3 months · 6 months · 1 year |
| **36 Regions** | All Indian States & Union Territories |
| **Daily Trend Line** | Segment-coloured, rolling average, peak/trough markers |
| **Momentum Banner** | Surging / Picking Up / Stable / Cooling Down / Dropping Fast |
| **8 Chart Types** | Bar · Map · Trend line · Heatmap · Radar · Bubble · Line · Box |
| **Demo Mode** | Instant synthetic data — no API calls needed |
| **Live Mode** | Real Google Trends data via pytrends |
| **CSV Export** | One-click download of full data table |

---

## 🚀 Deploy: GitHub → Streamlit Cloud

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/mosquito-trends-india.git
git push -u origin main
```

### 2. Connect Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Set:
   - **Repository:** `YOUR_USERNAME/mosquito-trends-india`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**

Your app will be live at `https://YOUR_USERNAME-mosquito-trends-india.streamlit.app`

---

## 🏃 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/mosquito-trends-india.git
cd mosquito-trends-india
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**

---

## 📂 Project Structure

```
mosquito-trends-india/
├── app.py                  ← Full Streamlit app (single file)
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml        ← Theme & server config
├── .gitignore
└── README.md
```

---

## 📈 Charts & Tabs

### Tab 1 — Period Analysis
| Chart | Description |
|---|---|
| **Top N Bar Chart** | Horizontal bars, colour-coded by score, for selected period |
| **India Choropleth** | Geographic heat map across all states |
| **Momentum Banner** | Full-width coloured banner showing trend direction & slope |
| **Daily Trend Line** | Segment-coloured line (🔴 above avg / 🔵 below avg), rolling average ribbon, peak ★ and trough ▼ markers, range slider |

### Tab 2 — Cross-Period View
| Chart | Description |
|---|---|
| **Heatmap** | All 36 states × 6 periods in a single grid |
| **Radar** | Top 8 states compared across all periods |
| **Box Plot** | Score distribution per period (median, IQR, outliers) |

### Tab 3 — Deep Dive
| Chart | Description |
|---|---|
| **Bubble Chart** | 7-day vs 1-year interest; bubble size = overall average |
| **Multi-state Line** | Compare any states across all time periods |

### Tab 4 — Data Table
- Colour-coded sortable table (all 36 states × 6 periods)
- One-click CSV download

---

## ⚠️ Trend Momentum Classification

| Status | Condition | Banner colour |
|---|---|---|
| 🚀 Surging | Recent avg ≥ +20% AND slope > 0.5 | Red |
| 📈 Picking Up | Recent avg ≥ +8% | Orange |
| ➡️ Stable | Within ±8% | Green |
| 🔽 Cooling Down | Recent avg ≤ −8% | Blue |
| 📉 Dropping Fast | Recent avg ≤ −20% AND slope < −0.5 | Dark blue |

---

## 🔧 Known Fixes

### urllib3 compatibility (`method_whitelist` error)
`pytrends` passes the old `method_whitelist` kwarg to `urllib3.Retry`, which was
renamed to `allowed_methods` in `urllib3 >= 1.26`.  
**Fix:** `app.py` monkey-patches `Retry.__init__` at module load to transparently
alias the old name, so pytrends works with any installed `urllib3` version.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pytrends` | Unofficial Google Trends API |
| `plotly` | Interactive charts |
| `pandas` | Data wrangling |
| `numpy` | Numerical ops |
| `urllib3<2` | HTTP — pinned to avoid breaking pytrends |

---

## 📄 License

MIT — free to use, modify, and deploy.

---

*Data: Google Trends. Scores are relative (0 = lowest, 100 = peak interest in region).
Not affiliated with Google.*
