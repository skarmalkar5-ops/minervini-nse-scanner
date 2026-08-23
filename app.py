
import io
import os
import smtplib
import time
from email.message import EmailMessage
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Minervini NSE Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Minervini NSE Scanner")
st.caption("NSE → Trend Template → RS → Leader Score → VCP proxy → CSV → Email")

MIN_TREND_SCORE = 88.9
MIN_RS_RATING = 80.0
MIN_LEADER_SCORE = 85.0
BATCH_SIZE = 100
PERIOD = "2y"

NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


# -----------------------------
# Helpers
# -----------------------------
def load_nse_universe():
    nse = pd.read_csv(NSE_URL)
    nse.columns = nse.columns.str.strip()
    nse_eq = (
        nse[nse["SERIES"] == "EQ"]
        .drop_duplicates(subset=["SYMBOL"])
        .copy()
    )
    nse_eq["YF_SYMBOL"] = nse_eq["SYMBOL"] + ".NS"
    return nse_eq


def clean_yf_batch(raw, symbol):
    if not isinstance(raw.columns, pd.MultiIndex):
        return None

    available = raw.columns.get_level_values(0).unique()
    if symbol not in available:
        return None

    stock = raw[symbol].copy()
    required = ["High", "Low", "Close", "Volume"]
    if not all(c in stock.columns for c in required):
        return None

    return stock[required].dropna(subset=["Close"])


def scan_trend(symbols):
    results = []

    raw = yf.download(
        symbols,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    for symbol in symbols:
        try:
            stock = clean_yf_batch(raw, symbol)
            if stock is None or len(stock) < 252:
                continue

            stock["50_DMA"] = stock["Close"].rolling(50).mean()
            stock["150_DMA"] = stock["Close"].rolling(150).mean()
            stock["200_DMA"] = stock["Close"].rolling(200).mean()

            stock = stock.dropna(subset=["50_DMA", "150_DMA", "200_DMA"])
            if len(stock) < 252:
                continue

            latest = stock.iloc[-1]
            price = float(latest["Close"])
            dma50 = float(latest["50_DMA"])
            dma150 = float(latest["150_DMA"])
            dma200 = float(latest["200_DMA"])
            dma200_20 = float(stock["200_DMA"].iloc[-21])

            high_52w = float(stock["High"].tail(252).max())
            low_52w = float(stock["Low"].tail(252).min())

            conditions = {
                "Price > 50 DMA": price > dma50,
                "Price > 150 DMA": price > dma150,
                "Price > 200 DMA": price > dma200,
                "50 DMA > 150 DMA": dma50 > dma150,
                "50 DMA > 200 DMA": dma50 > dma200,
                "150 DMA > 200 DMA": dma150 > dma200,
                "200 DMA rising": dma200 > dma200_20,
                "Near 52W High": price >= high_52w * 0.75,
                "Above 52W Low": price >= low_52w * 1.30,
            }

            passed = sum(conditions.values())

            results.append({
                "Symbol": symbol,
                "Price": round(price, 2),
                "50_DMA": round(dma50, 2),
                "150_DMA": round(dma150, 2),
                "200_DMA": round(dma200, 2),
                "52W_High": round(high_52w, 2),
                "52W_Low": round(low_52w, 2),
                "Passed": passed,
                "Trend_Score": round(passed / 9 * 100, 1),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def calculate_rs_batch(symbols, nifty_close):
    results = []

    raw = yf.download(
        symbols,
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()

    available = raw.columns.get_level_values(0).unique()

    for symbol in symbols:
        try:
            if symbol not in available:
                continue

            stock = raw[symbol]
            if "Close" not in stock.columns:
                continue

            stock_close = stock["Close"].dropna()
            if len(stock_close) < 252:
                continue

            combined = pd.concat(
                [stock_close, nifty_close],
                axis=1,
                join="inner",
            ).dropna()

            if len(combined) < 252:
                continue

            combined.columns = ["Stock", "NIFTY"]

            output = {"Symbol": symbol}
            for name, days in {"1M": 21, "3M": 63, "6M": 126, "12M": 252}.items():
                stock_return = (
                    combined["Stock"].iloc[-1]
                    / combined["Stock"].iloc[-days - 1]
                    - 1
                ) * 100

                nifty_return = (
                    combined["NIFTY"].iloc[-1]
                    / combined["NIFTY"].iloc[-days - 1]
                    - 1
                ) * 100

                output[f"{name}_Rel"] = round(stock_return - nifty_return, 2)

            results.append(output)
        except Exception:
            continue

    return pd.DataFrame(results)


def vcp_for_stock(stock):
    if len(stock) < 70:
        return None

    recent = stock.tail(60).copy()
    first_30 = recent.head(30)
    last_30 = recent.tail(30)

    range_first = (
        (first_30["High"].max() - first_30["Low"].min())
        / first_30["Close"].mean()
    )
    range_last = (
        (last_30["High"].max() - last_30["Low"].min())
        / last_30["Close"].mean()
    )
    range_contraction = range_last < range_first

    prev_close = recent["Close"].shift(1)
    tr1 = recent["High"] - recent["Low"]
    tr2 = (recent["High"] - prev_close).abs()
    tr3 = (recent["Low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()

    atr_early = atr.iloc[14:29].mean()
    atr_recent = atr.iloc[-15:].mean()
    atr_contraction = atr_recent < atr_early
    atr_change = (atr_recent / atr_early - 1) * 100 if atr_early else np.nan

    volume_first = first_30["Volume"].mean()
    volume_last = last_30["Volume"].mean()
    volume_contraction = volume_last < volume_first
    volume_change = (volume_last / volume_first - 1) * 100 if volume_first else np.nan

    high_60 = recent["High"].max()
    low_60 = recent["Low"].min()
    current_price = float(recent["Close"].iloc[-1])

    if high_60 == low_60:
        return None

    range_position = (current_price - low_60) / (high_60 - low_60)
    near_upper_range = range_position >= 0.70

    base_range_pct = ((high_60 - low_60) / current_price) * 100
    tight_base = base_range_pct <= 25

    passed = sum([
        range_contraction,
        atr_contraction,
        volume_contraction,
        near_upper_range,
        tight_base,
    ])

    return {
        "VCP Passed": passed,
        "VCP Score": round(passed / 5 * 100, 1),
        "Base Range %": round(base_range_pct, 2),
        "Range Position %": round(range_position * 100, 1),
        "ATR Change %": round(atr_change, 2),
        "Volume Change %": round(volume_change, 2),
    }


def send_email(csv_bytes, filename, row_count):
    # Configure these in Streamlit secrets.
    smtp_host = st.secrets["SMTP_HOST"]
    smtp_port = int(st.secrets.get("SMTP_PORT", 587))
    smtp_user = st.secrets["SMTP_USER"]
    smtp_password = st.secrets["SMTP_PASSWORD"]
    email_to = st.secrets["EMAIL_TO"]

    msg = EmailMessage()
    msg["Subject"] = f"Minervini NSE Scan — {datetime.now().strftime('%d-%b-%Y')}"
    msg["From"] = smtp_user
    msg["To"] = email_to
    msg.set_content(
        f"Minervini NSE scan completed.\n\n"
        f"Candidates in attached CSV: {row_count}\n"
        f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}\n"
    )
    msg.add_attachment(
        csv_bytes,
        maintype="text",
        subtype="csv",
        filename=filename,
    )

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def run_scan(progress):
    nse_eq = load_nse_universe()
    all_symbols = nse_eq["YF_SYMBOL"].tolist()

    progress(0.05, f"NSE universe loaded: {len(all_symbols)} stocks")

    nifty = yf.download(
        "^NSEI",
        period=PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty_close = nifty["Close"].dropna()

    trend_parts = []
    rs_parts = []

    total_batches = (len(all_symbols) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(all_symbols), BATCH_SIZE):
        batch = all_symbols[i:i + BATCH_SIZE]
        batch_no = i // BATCH_SIZE + 1

        progress(
            0.05 + 0.55 * (batch_no / total_batches),
            f"Scanning batch {batch_no}/{total_batches}…",
        )

        trend_batch = scan_trend(batch)
        rs_batch = calculate_rs_batch(batch, nifty_close)

        if not trend_batch.empty:
            trend_parts.append(trend_batch)
        if not rs_batch.empty:
            rs_parts.append(rs_batch)

        time.sleep(0.5)

    trend_df = pd.concat(trend_parts, ignore_index=True)
    rs_df = pd.concat(rs_parts, ignore_index=True)

    leaderboard = trend_df.merge(rs_df, on="Symbol", how="inner")

    for period in ["1M_Rel", "3M_Rel", "6M_Rel", "12M_Rel"]:
        leaderboard[f"{period}_Rank"] = (
            leaderboard[period].rank(pct=True) * 100
        )

    leaderboard["RS_Rating"] = (
        leaderboard["1M_Rel_Rank"] * 0.15
        + leaderboard["3M_Rel_Rank"] * 0.25
        + leaderboard["6M_Rel_Rank"] * 0.25
        + leaderboard["12M_Rel_Rank"] * 0.35
    ).round(1)

    leaderboard["Leader_Score"] = (
        leaderboard["Trend_Score"] * 0.50
        + leaderboard["RS_Rating"] * 0.50
    ).round(1)

    candidates = leaderboard[
        (leaderboard["Trend_Score"] >= MIN_TREND_SCORE)
        & (leaderboard["RS_Rating"] >= MIN_RS_RATING)
        & (leaderboard["Leader_Score"] >= MIN_LEADER_SCORE)
    ].copy()

    progress(0.65, f"Leader filter: {len(candidates)} candidates")

    # VCP analysis only for candidates.
    candidate_symbols = candidates["Symbol"].tolist()
    vcp_parts = []

    for i in range(0, len(candidate_symbols), BATCH_SIZE):
        batch = candidate_symbols[i:i + BATCH_SIZE]
        batch_no = i // BATCH_SIZE + 1
        total_vcp_batches = max(1, (len(candidate_symbols) + BATCH_SIZE - 1) // BATCH_SIZE)

        progress(
            0.65 + 0.30 * (batch_no / total_vcp_batches),
            f"VCP analysis {batch_no}/{total_vcp_batches}…",
        )

        raw = yf.download(
            batch,
            period="1y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        for symbol in batch:
            try:
                stock = clean_yf_batch(raw, symbol)
                if stock is None:
                    continue
                v = vcp_for_stock(stock)
                if v is not None:
                    vcp_parts.append({"Symbol": symbol, **v})
            except Exception:
                continue

    vcp_df = pd.DataFrame(vcp_parts)
    final = candidates.merge(vcp_df, on="Symbol", how="left")

    final = final[
        final["VCP Score"].fillna(0) >= 80
    ].copy()

    final = final.sort_values(
        ["Leader_Score", "VCP Score"],
        ascending=False,
    ).reset_index(drop=True)

    final["Symbol"] = final["Symbol"].str.replace(".NS", "", regex=False)

    progress(1.0, f"Scan complete: {len(final)} final candidates")

    return final, len(all_symbols), len(trend_df), len(leaderboard)


# -----------------------------
# UI
# -----------------------------
st.sidebar.header("Scanner rules")
st.sidebar.write(f"Trend Score ≥ {MIN_TREND_SCORE}")
st.sidebar.write(f"RS Rating ≥ {MIN_RS_RATING}")
st.sidebar.write(f"Leader Score ≥ {MIN_LEADER_SCORE}")
st.sidebar.write("VCP Score ≥ 80")
st.sidebar.caption("No arbitrary top-50 limit.")

if "results" not in st.session_state:
    st.session_state.results = None

if st.button("🔄 REFRESH SCAN", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status = st.empty()

    def update_progress(value, message):
        progress_bar.progress(min(max(value, 0.0), 1.0))
        status.info(message)

    try:
        with st.spinner("Running full NSE scan…"):
            final, universe_count, trend_count, leader_count = run_scan(update_progress)

        st.session_state.results = final
        st.session_state.scan_meta = {
            "universe": universe_count,
            "trend": trend_count,
            "leaders": leader_count,
        }

        st.success("Scan completed successfully.")

        csv_bytes = final.to_csv(index=False).encode("utf-8")
        filename = f"minervini_scan_{datetime.now().strftime('%Y-%m-%d')}.csv"

        try:
            send_email(csv_bytes, filename, len(final))
            st.success("📧 CSV emailed successfully.")
        except Exception as email_error:
            st.warning(
                "Scan completed, but email could not be sent. "
                "Check your SMTP secrets."
            )
            st.exception(email_error)

    except Exception as e:
        st.error("The scan failed.")
        st.exception(e)

if st.session_state.results is not None:
    final = st.session_state.results
    meta = st.session_state.scan_meta

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NSE EQ", meta["universe"])
    c2.metric("Usable Trend", meta["trend"])
    c3.metric("Leader Pool", meta["leaders"])
    c4.metric("Final VCP", len(final))

    st.subheader("Final candidates")
    st.dataframe(final, use_container_width=True, hide_index=True)

    csv_bytes = final.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        data=csv_bytes,
        file_name=f"minervini_scan_{datetime.now().strftime('%Y-%m-%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
