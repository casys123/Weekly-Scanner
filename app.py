# Retry writing the Streamlit CSP/CC scanner project files and zipping them.
import os, zipfile
from pathlib import Path

project_dir = Path("/mnt/data/CSP_CC_Scanner_Streamlit")
project_dir.mkdir(parents=True, exist_ok=True)

app_py = '''
import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone
from io import BytesIO
from scipy.stats import norm

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Weekly CSP & Covered Call Scanner", page_icon="📊", layout="wide")
st.title("📊 Weekly CSP & Covered Call Scanner")
st.caption("Scan for Cash‑Secured Puts and Covered Calls (7–14 DTE) with capital and mid‑cap filters. Data via Yahoo Finance (yfinance).")

with st.sidebar:
    st.header("Controls")
    default_tickers = "KO,PFE,F,SOFI,XLF,T,BAC,WFC,KHC,SNAP,GM,CSCO,INTC,KMI,C,PBR,VALE,PARA,NOK,CCL"
    tickers_text = st.text_area("Tickers (comma‑separated)", value=default_tickers, height=90)
    capital = st.number_input("Max capital per contract ($)", min_value=500.0, value=4000.0, step=250.0, format="%.0f")
    dte_min = st.number_input("Min DTE", min_value=1, value=7, step=1)
    dte_max = st.number_input("Max DTE", min_value=1, value=14, step=1)
    delta_min = st.number_input("Abs Δ min", min_value=0.05, value=0.30, step=0.05, format="%.2f")
    delta_max = st.number_input("Abs Δ max", min_value=0.10, value=0.40, step=0.05, format="%.2f")
    min_return = st.number_input("Min return (% of strike or notional)", min_value=0.1, value=1.0, step=0.1, format="%.1f") / 100.0
    min_oi = st.number_input("Min Open Interest", min_value=0, value=50, step=10)
    min_vol = st.number_input("Min Volume", min_value=0, value=10, step=10)
    price_cap_for_CC = st.number_input("Max underlying price for Covered Calls (100 shares within capital)", min_value=1.0, value=40.0, step=1.0, format="%.2f")
    allow_cc_ignore_cap = st.checkbox("Covered Calls: ignore capital constraint (I already own shares)", value=False)
    st.divider()
    midcap_only = st.checkbox("Filter to Mid‑Cap (Market Cap $2B–$10B)", value=False)
    st.caption("Tip: Mid‑cap window is approximate; data source is Yahoo Finance.")

def annual_rf_rate():
    return 0.045

def put_delta(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        return norm.cdf(d1) - 1.0
    except Exception:
        return np.nan

def call_delta(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        return norm.cdf(d1)
    except Exception:
        return np.nan

def to_annualized_time(dte):
    return max(1e-8, dte / 365.0)

@st.cache_data(show_spinner=False)
def get_info_for_ticker(ticker: str):
    if yf is None:
        return None
    t = yf.Ticker(ticker)
    price = None
    mcap = None
    try:
        fi = t.fast_info
        price = float(fi.get("last_price", np.nan))
        mcap = float(fi.get("market_cap", np.nan))
    except Exception:
        pass
    if (price is None) or (price != price) or (price <= 0):
        try:
            price = float(t.history(period="1d")["Close"].iloc[-1])
        except Exception:
            price = np.nan
    if (mcap is None) or (mcap != mcap) or (mcap <= 0):
        try:
            mcap = float(t.info.get("marketCap", np.nan))
        except Exception:
            mcap = np.nan
    return {"price": price, "market_cap": mcap, "ticker": ticker, "expirations": t.options or [], "t": t}

def within_midcap(mcap):
    try:
        return (mcap >= 2e9) and (mcap <= 1e10)
    except Exception:
        return False

def scan_csp_for_ticker(tinfo, cfg):
    t = tinfo["t"]; price = tinfo["price"]; mcap = tinfo["market_cap"]
    outs = []
    if not np.isfinite(price) or price <= 0:
        return pd.DataFrame()
    today = datetime.now(timezone.utc).date()
    for exp in tinfo["expirations"]:
        try:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (d - today).days
            if not (cfg["dte_min"] <= dte <= cfg["dte_max"]):
                continue
            chain = t.option_chain(exp)
            puts = chain.puts.copy()
            if puts.empty:
                continue
            puts["mid"] = (puts["bid"].fillna(0) + puts["ask"].fillna(0)) / 2.0
            puts["mid"] = puts["mid"].mask(puts["mid"] <= 0, puts["lastPrice"].fillna(0))
            puts["premium"] = puts["mid"]
            puts["collateral"] = puts["strike"] * 100.0
            T = to_annualized_time(dte)
            rf = annual_rf_rate()
            iv = puts["impliedVolatility"].astype(float).replace([np.inf, -np.inf], np.nan)
            iv = iv.mask(iv <= 0, 0.25)
            deltas = []
            for K, sigma in zip(puts["strike"].values.astype(float), iv.values.astype(float)):
                deltas.append(put_delta(price, K, T, rf, sigma))
            puts["delta"] = deltas
            puts["prem_perc_of_strike"] = (puts["premium"] / puts["strike"]).replace([np.inf, -np.inf], np.nan)
            mask = (
                (puts["collateral"] <= cfg["capital"]) &
                (puts["openInterest"].fillna(0) >= cfg["min_oi"]) &
                (puts["volume"].fillna(0) >= cfg["min_vol"]) &
                (puts["delta"].abs().between(cfg["delta_min"], cfg["delta_max"])) &
                (puts["prem_perc_of_strike"] >= cfg["min_return"])
            )
            sel = puts.loc[mask, ["contractSymbol","strike","bid","ask","mid","premium","openInterest","volume","impliedVolatility","delta"]].copy()
            if sel.empty:
                continue
            sel["ticker"] = tinfo["ticker"]
            sel["underlying_price"] = price
            sel["market_cap"] = mcap
            sel["expiration"] = exp
            sel["dte"] = dte
            outs.append(sel)
        except Exception:
            continue
    if not outs:
        return pd.DataFrame()
    out = pd.concat(outs, ignore_index=True)
    out["iv"] = (out["impliedVolatility"] * 100).round(2)
    out["prem_perc_of_strike"] = (out["premium"] / out["strike"] * 100).round(2)
    out["delta"] = out["delta"].round(3)
    out["underlying_price"] = out["underlying_price"].round(2)
    out.sort_values(["prem_perc_of_strike","volume","openInterest"], ascending=[False, False, False], inplace=True)
    return out[["ticker","underlying_price","market_cap","expiration","dte","strike","premium","prem_perc_of_strike","delta","iv","openInterest","volume","bid","ask","mid","contractSymbol"]]

def scan_cc_for_ticker(tinfo, cfg):
    t = tinfo["t"]; price = tinfo["price"]; mcap = tinfo["market_cap"]
    outs = []
    if not np.isfinite(price) or price <= 0:
        return pd.DataFrame()
    if (not cfg["cc_ignore_cap"]) and (price * 100.0 > cfg["capital"]):
        return pd.DataFrame()
    if price > cfg["cc_price_cap"] and not cfg["cc_ignore_cap"]:
        return pd.DataFrame()
    today = datetime.now(timezone.utc).date()
    for exp in tinfo["expirations"]:
        try:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (d - today).days
            if not (cfg["dte_min"] <= dte <= cfg["dte_max"]):
                continue
            chain = t.option_chain(exp)
            calls = chain.calls.copy()
            if calls.empty:
                continue
            calls["mid"] = (calls["bid"].fillna(0) + calls["ask"].fillna(0)) / 2.0
            calls["mid"] = calls["mid"].mask(calls["mid"] <= 0, calls["lastPrice"].fillna(0))
            calls["premium"] = calls["mid"]
            T = to_annualized_time(dte)
            rf = annual_rf_rate()
            iv = calls["impliedVolatility"].astype(float).replace([np.inf, -np.inf], np.nan)
            iv = iv.mask(iv <= 0, 0.25)
            deltas = []
            for K, sigma in zip(calls["strike"].values.astype(float), iv.values.astype(float)):
                deltas.append(call_delta(price, K, T, rf, sigma))
            calls["delta"] = deltas
            calls["prem_perc_of_notional"] = (calls["premium"] / price / 100.0 * 10000.0)  # %
            mask = (
                (calls["openInterest"].fillna(0) >= cfg["min_oi"]) &
                (calls["volume"].fillna(0) >= cfg["min_vol"]) &
                (calls["delta"].between(cfg["delta_min"], cfg["delta_max"])) &
                (calls["prem_perc_of_notional"] >= (cfg["min_return"]*100))
            )
            sel = calls.loc[mask, ["contractSymbol","strike","bid","ask","mid","premium","openInterest","volume","impliedVolatility","delta"]].copy()
            if sel.empty:
                continue
            sel["ticker"] = tinfo["ticker"]
            sel["underlying_price"] = price
            sel["market_cap"] = mcap
            sel["expiration"] = exp
            sel["dte"] = dte
            outs.append(sel)
        except Exception:
            continue
    if not outs:
        return pd.DataFrame()
    out = pd.concat(outs, ignore_index=True)
    out["iv"] = (out["impliedVolatility"] * 100).round(2)
    out["prem_perc_of_notional"] = out["prem_perc_of_notional"].round(2)
    out["delta"] = out["delta"].round(3)
    out["underlying_price"] = out["underlying_price"].round(2)
    out.sort_values(["prem_perc_of_notional","volume","openInterest"], ascending=[False, False, False], inplace=True)
    return out[["ticker","underlying_price","market_cap","expiration","dte","strike","premium","prem_perc_of_notional","delta","iv","openInterest","volume","bid","ask","mid","contractSymbol"]]

# Parse tickers
tickers = [s.strip().upper() for s in tickers_text.split(",") if s.strip()]
cfg = dict(
    capital=capital, dte_min=dte_min, dte_max=dte_max,
    delta_min=delta_min, delta_max=delta_max, min_return=min_return,
    min_oi=min_oi, min_vol=min_vol, cc_price_cap=price_cap_for_CC, cc_ignore_cap=allow_cc_ignore_cap
)

if yf is None:
    st.error("yfinance is not available. When you deploy or run locally with internet, the app will fetch live chains.")
    st.stop()

info_rows = []
for tk in tickers:
    tinfo = get_info_for_ticker(tk)
    if tinfo is None:
        continue
    if midcap_only and not within_midcap(tinfo["market_cap"]):
        continue
    info_rows.append(tinfo)

if not info_rows:
    st.warning("No tickers passed the initial filters (mid‑cap filter might be too strict). Try unchecking mid‑cap or adding more tickers.")
else:
    st.write(f"Scanning **{len(info_rows)}** tickers…")

csp_tables = []
cc_tables = []

for tinfo in info_rows:
    csp_df = scan_csp_for_ticker(tinfo, cfg)
    if not csp_df.empty:
        csp_tables.append(csp_df)
    cc_df = scan_cc_for_ticker(tinfo, cfg)
    if not cc_df.empty:
        cc_tables.append(cc_df)

st.markdown("### Cash‑Secured Put Candidates")
if csp_tables:
    csp_all = pd.concat(csp_tables, ignore_index=True)
    st.dataframe(csp_all, use_container_width=True)
    st.caption(f"Found {len(csp_all)} CSP rows.")
    csp_csv = csp_all.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSP CSV", csp_csv, "csp_candidates.csv", "text/csv")
else:
    st.info("No CSP candidates matched your filters. Try widening delta or return %.")

st.markdown("### Covered Call Candidates")
if cc_tables:
    cc_all = pd.concat(cc_tables, ignore_index=True)
    st.dataframe(cc_all, use_container_width=True)
    st.caption(f"Found {len(cc_all)} CC rows.")
    cc_csv = cc_all.to_csv(index=False).encode("utf-8")
    st.download_button("Download CC CSV", cc_csv, "cc_candidates.csv", "text/csv")
else:
    st.info("No Covered Call candidates matched your filters. Try raising price cap or ignoring capital constraint if you already own shares.")

st.markdown("---")
st.subheader("Export Ranked HTML Report")
def df_to_html_download(csp_df: pd.DataFrame, cc_df: pd.DataFrame) -> bytes:
    html_parts = ["<html><head><meta charset='utf-8'><title>Weekly CSP/CC Report</title>",
                  "<style>body{font-family:Arial; padding:20px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ccc;padding:6px;} th{background:#f3f3f3;}</style></head><body>",
                  "<h1>Weekly CSP & Covered Call Report</h1>",
                  f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"]
    if csp_df is not None and not csp_df.empty:
        html_parts.append("<h2>Cash‑Secured Puts</h2>")
        html_parts.append(csp_df.to_html(index=False, escape=False))
    else:
        html_parts.append("<h2>Cash‑Secured Puts</h2><p>No rows.</p>")
    if cc_df is not None and not cc_df.empty:
        html_parts.append("<h2>Covered Calls</h2>")
        html_parts.append(cc_df.to_html(index=False, escape=False))
    else:
        html_parts.append("<h2>Covered Calls</h2><p>No rows.</p>")
    html_parts.append("</body></html>")
    return "\n".join(html_parts).encode("utf-8")

csp_all_for_html = pd.concat(csp_tables, ignore_index=True) if csp_tables else pd.DataFrame()
cc_all_for_html = pd.concat(cc_tables, ignore_index=True) if cc_tables else pd.DataFrame()
html_bytes = df_to_html_download(csp_all_for_html, cc_all_for_html)
st.download_button("Download HTML Report", data=html_bytes, file_name="weekly_csp_cc_report.html", mime="text/html")

st.markdown("---")
st.caption("DISCLAIMER: Educational purposes only. Not investment advice. Verify all quotes/greeks in your broker (Thinkorswim/Tastytrade).")
'''

readme_md = '''
# Weekly CSP & Covered Call Scanner — Streamlit

A Streamlit app that scans **Cash‑Secured Puts (CSPs)** and **Covered Calls (CCs)** with short **7–14 DTE**, tuned for a capital limit (e.g., **$4,000**). Includes an optional **Mid‑Cap filter ($2B–$10B)** and exports **CSV** and **HTML** ranked reports.

**Data Source**: Yahoo Finance via `yfinance` (delayed; verify in your broker).

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
