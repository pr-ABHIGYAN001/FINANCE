"""
Quantitative Finance Dashboard
================================
A Streamlit-based interactive dashboard that:
  1. Fetches live stock data via yfinance.
  2. Engineers technical features (SMA, returns, volatility).
  3. Visualises price trends with Matplotlib.
  4. Trains a Logistic Regression model to predict the next day's direction.

Author : AI-assisted build
Stack  : Python · Streamlit · yfinance · Pandas · NumPy · Scikit-Learn · Matplotlib
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0 ▸ IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import datetime

import matplotlib.pyplot as plt          # Plotting library
import numpy as np                       # Numerical computing
import pandas as pd                      # Tabular data manipulation
import streamlit as st                   # Web-app framework
import yfinance as yf                    # Yahoo Finance market-data API
from sklearn.linear_model import LogisticRegression  # Binary classifier
from sklearn.metrics import accuracy_score           # Model evaluation

# ──────────────────────────────────────────────────────────────────────────────
# 1 ▸ STREAMLIT PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# `set_page_config` MUST be the very first Streamlit call in the script.
# It controls the browser tab title, favicon, and initial layout width.
st.set_page_config(
    page_title="Quant Finance Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Main heading & intro text ────────────────────────────────────────────────
st.title("📈 Quantitative Finance Dashboard")
st.markdown(
    "Fetch live market data, visualise price trends with moving averages, "
    "and let a **Logistic Regression** model predict the next trading day's "
    "direction — all from a single interactive page."
)

# ──────────────────────────────────────────────────────────────────────────────
# 2 ▸ SIDEBAR — USER INPUTS
# ──────────────────────────────────────────────────────────────────────────────
# The sidebar keeps configuration controls out of the main content area,
# giving the dashboard a clean, report-like feel.

st.sidebar.header("⚙️ Settings")

# ── Ticker ────────────────────────────────────────────────────────────────────
# A plain text input so users can type any valid Yahoo Finance ticker symbol.
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

# ── Date range ────────────────────────────────────────────────────────────────
# Default: two years of history — enough for the 200-day SMA to have a
# meaningful run-in period before it becomes available.
today = datetime.date.today()
default_start = today - datetime.timedelta(days=2 * 365)  # ~2 years ago

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

# Guard: start must come before end.
if start_date >= end_date:
    st.sidebar.error("Start Date must be earlier than End Date.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 3 ▸ DATA ACQUISITION (yfinance)
# ──────────────────────────────────────────────────────────────────────────────
# `yf.download` returns a DataFrame indexed by date with columns:
#   Open, High, Low, Close, Adj Close, Volume.
# We pass `progress=False` to suppress the download bar in Streamlit logs.

st.subheader(f"🔍 Fetching data for **{ticker}** …")

raw_df = yf.download(ticker, start=start_date, end=end_date, progress=False)

# yfinance may return a MultiIndex if multiple tickers are passed (even with
# a single ticker in some versions).  We flatten it so column access like
# df["Close"] always works.
if isinstance(raw_df.columns, pd.MultiIndex):
    raw_df.columns = raw_df.columns.get_level_values(0)

# If the download returned nothing the ticker is probably invalid.
if raw_df.empty:
    st.error(
        f"No data returned for **{ticker}**. "
        "Please check the ticker symbol and date range."
    )
    st.stop()

# Work on a copy so `raw_df` stays untouched for reference.
df = raw_df.copy()

# ──────────────────────────────────────────────────────────────────────────────
# 4 ▸ FEATURE ENGINEERING (Pandas & NumPy)
# ──────────────────────────────────────────────────────────────────────────────
# We create four technical columns that will later serve as ML features,
# plus one Target column for the model to learn from.

# ── 4a. Simple Moving Averages (SMA) ────────────────────────────────────────
# The SMA smooths price noise by averaging the closing price over the last
# N days.  The 50-day and 200-day SMAs are institutional standards:
#   - Price above SMA_200 → long-term uptrend.
#   - SMA_50 crossing above SMA_200 ("golden cross") → bullish signal.
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()

# ── 4b. Daily Return ────────────────────────────────────────────────────────
# Percentage change from one close to the next: (P_t - P_{t-1}) / P_{t-1}.
# This is the most fundamental measure of single-day performance.
df["Daily_Return"] = df["Close"].pct_change()

# ── 4c. Annualized Volatility ───────────────────────────────────────────────
# Volatility = 20-day rolling standard deviation of daily returns, then
# scaled to an annual figure by multiplying by √252 (≈ trading days / year).
# Higher volatility → more risk / larger expected price swings.
df["Volatility"] = df["Daily_Return"].rolling(window=20).std() * np.sqrt(252)

# ── 4d. ML Target ───────────────────────────────────────────────────────────
# Binary label for supervised learning:
#   1 → tomorrow's close is STRICTLY higher than today's (UP day).
#   0 → tomorrow's close is equal to or lower (DOWN day).
# We use `shift(-1)` to look one row into the future.
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# ── 4e. Drop NaN rows ───────────────────────────────────────────────────────
# Rolling windows (50, 200) and shift(-1) introduce NaN values at the edges.
# Dropping them ensures clean inputs for both plotting and modelling.
df.dropna(inplace=True)

# Quick sanity check: after dropping NaNs we still need enough rows to
# train and test a model.
if len(df) < 30:
    st.error(
        "Not enough data rows after feature engineering. "
        "Try a wider date range or a different ticker."
    )
    st.stop()

# Show the most recent rows so the user can verify the data.
with st.expander("📋 Preview Engineered Data (last 10 rows)", expanded=False):
    st.dataframe(df.tail(10), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5 ▸ VISUALIZATION (Matplotlib)
# ──────────────────────────────────────────────────────────────────────────────
# A single chart showing the closing price overlaid with both SMAs gives an
# at-a-glance view of trend and momentum.

st.subheader("📊 Price & Moving Averages")

fig, ax = plt.subplots(figsize=(14, 5))

# Plot lines with distinct styles so they're easy to tell apart.
ax.plot(df.index, df["Close"], label="Close", linewidth=1.4, color="#1f77b4")
ax.plot(df.index, df["SMA_50"], label="SMA 50", linewidth=1.1,
        linestyle="--", color="#ff7f0e")
ax.plot(df.index, df["SMA_200"], label="SMA 200", linewidth=1.1,
        linestyle="--", color="#2ca02c")

ax.set_title(f"{ticker} — Closing Price with 50 & 200-day SMA", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()

# `st.pyplot` embeds the Matplotlib figure directly into the Streamlit page.
st.pyplot(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 6 ▸ MACHINE LEARNING PIPELINE (Scikit-Learn)
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("🤖 Machine Learning — Next-Day Direction Prediction")

# ── 6a. Define features & target ────────────────────────────────────────────
# These four columns capture trend (SMAs), momentum (return), and risk
# (volatility) — a minimal but meaningful feature set.
feature_cols = ["SMA_50", "SMA_200", "Daily_Return", "Volatility"]
X = df[feature_cols]
y = df["Target"]

# ── 6b. Chronological train / test split ────────────────────────────────────
# TIME-SERIES RULE: Never shuffle! Future data must not leak into training.
# We use the first 80 % of rows for training and the remaining 20 % for
# testing, preserving temporal order.
split_idx = int(len(df) * 0.80)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ── 6c. Train the model ─────────────────────────────────────────────────────
# Logistic Regression is a simple, interpretable classifier ideal for binary
# outcomes (UP / DOWN).  `max_iter=1000` gives the solver enough room to
# converge even on larger feature sets.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ── 6d. Evaluate on the test set ────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy prominently.
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Model Accuracy (Test Set)",
        value=f"{accuracy:.2%}",
        help="Fraction of correctly predicted UP / DOWN days on the "
             "held-out test set (last 20 % of the time series).",
    )

with col2:
    st.metric(
        label="Test Set Size",
        value=f"{len(y_test)} days",
        help="Number of trading days used for evaluation.",
    )

# ── 6e. Predict the next trading day ────────────────────────────────────────
# We feed the most recent row of features into the trained model.
# This simulates asking: "Given today's technical indicators, will
# tomorrow's close be higher or lower?"
latest_features = X.iloc[[-1]]            # Keep as DataFrame for sklearn
next_day_pred = model.predict(latest_features)[0]

st.markdown("---")
st.subheader("🔮 Prediction for the Next Trading Day")

if next_day_pred == 1:
    st.success(
        f"📈 The model predicts **{ticker}** will close **UP** "
        "on the next trading day."
    )
else:
    st.error(
        f"📉 The model predicts **{ticker}** will close **DOWN** "
        "on the next trading day."
    )

# ── Disclaimer ───────────────────────────────────────────────────────────────
st.caption(
    "⚠️ *This dashboard is for educational purposes only and does NOT "
    "constitute financial advice. Past performance does not guarantee "
    "future results.*"
)
