# 🦟 Mosquito Google Trends — India State Analysis

An interactive Streamlit dashboard that visualises Google Trends search interest for **"Mosquito"** across all 36 Indian States & Union Territories for six time windows.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📸 Features

| Feature | Detail |
|---|---|
| **6 Time Periods** | 7 days · 15 days · 30 days · 3 months · 6 months · 1 year |
| **36 Regions** | All Indian States & Union Territories |
| **8 Chart Types** | Bar · Map · Time-series · Heatmap · Radar · Bubble · Line · Box |
| **Demo Mode** | Works instantly with synthetic data — no API key needed |
| **Live Mode** | Pulls real data from Google Trends via `pytrends` |
| **CSV Export** | Download the full data table with one click |

---

## 🚀 Deploy: GitHub → Streamlit Cloud (5 steps)

### Step 1 — Fork / push to GitHub

```bash
git clone https://github.com/YOUR_USERNAME/mosquito-trends-india.git
cd mosquito-trends-india
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2 — Sign in to Streamlit Cloud

Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.

### Step 3 — Create a new app

Click **"New app"** and fill in:

| Field | Value |
|---|---|
| **Repository** | `YOUR_USERNAME/mosquito-trends-india` |
| **Branch** | `main` |
| **Main file path** | `app.py` |

### Step 4 — Deploy 🎉

Click **"Deploy!"** — Streamlit Cloud will install `requirements.txt` and launch your app automatically.  
You'll get a public URL like `https://mosquito-trends-india.streamlit.app`.

### Step 5 — (Optional) Enable Live Mode

By default the app runs in **Demo Mode** (synthetic data, no API calls).  
To use live Google Trends data, simply toggle **"Demo Mode"** off in the sidebar.

> ⚠️ Google Trends rate-limits heavy usage. If you hit errors in Live mode, re-enable Demo Mode and try again later.

---

## 🗂️ Project Structure

```
mosquito-trends-india/
├── app.py                  ← Main Streamlit app (single file)
├── requirements.txt        ← Dependencies for Streamlit Cloud
├── .streamlit/
│   └── config.toml        ← Theme & server config
├── .gitignore
└── README.md
```

---

## 📊 Charts Explained

### Tab 1 — Period Analysis
- **Horizontal Bar Chart** — Top N states for the selected period, colour-coded by score
- **India Choropleth Map** — Geographic heat map of search interest across states
- **Time-series Line Chart** — National interest trend over the selected period with rolling average

### Tab 2 — Cross-Period View
- **Heatmap** — All 36 states × 6 periods in a single colour-coded grid
- **Radar Chart** — Top 8 states compared simultaneously across all periods
- **Box Plot** — Score distribution per period (median, IQR, outliers)

### Tab 3 — Deep Dive
- **Bubble Chart** — Short-term (7-day) vs long-term (1-year) interest; bubble size = overall average
- **Multi-state Line Chart** — Compare any set of states across all time periods

### Tab 4 — Data Table
- Colour-coded sortable table with all 36 states × 6 periods
- One-click **CSV download**

---

## 🏃 Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/mosquito-trends-india.git
cd mosquito-trends-india

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pytrends` | Unofficial Google Trends API |
| `plotly` | Interactive charts & choropleth maps |
| `pandas` | Data wrangling |
| `numpy` | Numerical operations |

---

## 🌍 Covered States & Union Territories

All 28 States + 8 UTs:  
Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Goa, Gujarat, Haryana,
Himachal Pradesh, Jharkhand, Karnataka, Kerala, Madhya Pradesh, Maharashtra, Manipur,
Meghalaya, Mizoram, Nagaland, Odisha, Punjab, Rajasthan, Sikkim, Tamil Nadu, Telangana,
Tripura, Uttar Pradesh, Uttarakhand, West Bengal, Delhi, Jammu & Kashmir, Ladakh,
Chandigarh, Puducherry, Andaman & Nicobar, Lakshadweep, Dadra & Nagar Haveli.

---

## 📄 License

MIT — free to use, modify, and deploy.

---

*Data: Google Trends. Scores are relative (0 = no interest, 100 = peak interest in region). This project is not affiliated with Google.*
