# 📈 Quantitative Finance Dashboard

An interactive Streamlit dashboard that fetches live stock data, performs technical analysis, and uses a Logistic Regression model to predict the next trading day's direction.

## Features

- **Live Data** — Pulls real market data from Yahoo Finance via `yfinance`.
- **Technical Indicators** — 50-day & 200-day Simple Moving Averages, Daily Returns, Annualized Volatility.
- **Interactive Chart** — Matplotlib visualisation of price trends with SMA overlays.
- **ML Prediction** — Logistic Regression trained on a chronological split predicts UP / DOWN for the next trading day.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/pr-ABHIGYAN001/FINANCE.git
cd FINANCE

# 2. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

## Tech Stack

Python · Streamlit · yfinance · Pandas · NumPy · Scikit-Learn · Matplotlib

## Disclaimer

⚠️ This project is for **educational purposes only** and does NOT constitute financial advice.
