"""
Dynamic Universe Builder
Combines: Top 100 most traded + S&P 500 + Most volatile of the day
Runs fresh each bot cycle to catch momentum stocks
"""

import logging
import time
import requests
import pandas as pd
import yfinance as yf
from typing import List

log = logging.getLogger(__name__)

# ── Fallback hardcoded list if dynamic fetch fails ────────────────────────────
FALLBACK_UNIVERSE = [
    "SPY", "QQQ", "IWM", "VTI", "DIA", "XLK", "XLV", "XLE", "XLF", "XLY",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC",
    "GS", "JNJ", "UNH", "PFE", "COST", "WMT", "HD", "XOM", "CVX", "AMD",
    "COIN", "PLTR", "SOFI", "UBER", "ABNB", "SHOP", "TQQQ", "SOXL", "ARKK",
    "MU", "INTC", "CRM", "NFLX", "DIS", "BA", "CAT", "MMM", "GE", "F", "GM",
]


def get_sp500() -> List[str]:
    """Fetch all S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # Clean up tickers (some have dots like BRK.B -> BRK-B for yfinance)
        tickers = [t.replace(".", "-") for t in tickers]
        log.info(f"✅ S&P 500: fetched {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(f"S&P 500 fetch failed: {e}")
        return []


def get_most_traded() -> List[str]:
    """Get top 100 most actively traded US stocks today via Yahoo Finance screener."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            "scrIds": "most_actives",
            "count":  100,
            "lang":   "en-US",
            "region": "US",
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        quotes = data["finance"]["result"][0]["quotes"]
        tickers = [q["symbol"] for q in quotes if "." not in q["symbol"]]
        log.info(f"✅ Most traded: fetched {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(f"Most traded fetch failed: {e}")
        return []


def get_most_volatile() -> List[str]:
    """Get top gainers and losers today — highest volatility = most opportunity."""
    tickers = []
    try:
        for scrId in ["day_gainers", "day_losers"]:
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            params = {"scrIds": scrId, "count": 50, "lang": "en-US", "region": "US"}
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, params=params, headers=headers, timeout=10)
            data = r.json()
            quotes = data["finance"]["result"][0]["quotes"]
            batch = [q["symbol"] for q in quotes if "." not in q["symbol"]]
            tickers.extend(batch)
            time.sleep(0.5)
        log.info(f"✅ Most volatile: fetched {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(f"Most volatile fetch failed: {e}")
        return []


def filter_tradeable(tickers: List[str], max_symbols: int = 300) -> List[str]:
    """
    Filter out untradeable symbols:
    - Remove warrants, preferred shares, ETNs (suffixes like W, WS, U, R)
    - Remove very short tickers that are often illiquid
    - Deduplicate and cap at max_symbols
    """
    seen  = set()
    clean = []
    bad_suffixes = {"W", "WS", "U", "R", "P", "Q"}

    for t in tickers:
        t = t.strip().upper()
        if not t or t in seen:
            continue
        # Skip warrants/preferred (e.g. SPAC.WS, ACMR.U)
        if any(t.endswith(s) for s in bad_suffixes):
            continue
        # Skip very long tickers (>5 chars usually illiquid)
        if len(t) > 5:
            continue
        seen.add(t)
        clean.append(t)
        if len(clean) >= max_symbols:
            break

    return clean


def build_universe() -> List[str]:
    """
    Build the combined smart universe:
    1. S&P 500 (stable, liquid)
    2. Most traded today (momentum)
    3. Most volatile today (opportunity)
    4. Deduplicate + filter
    5. Cap at 300 for performance
    """
    log.info("🌍 Building dynamic universe...")

    all_tickers = []

    # Priority order: volatile first (most opportunity), then traded, then S&P
    volatile = get_most_volatile()
    all_tickers.extend(volatile)
    time.sleep(0.5)

    traded = get_most_traded()
    all_tickers.extend(traded)
    time.sleep(0.5)

    sp500 = get_sp500()
    all_tickers.extend(sp500)

    # Filter and deduplicate
    universe = filter_tradeable(all_tickers, max_symbols=300)

    if len(universe) < 20:
        log.warning("Dynamic universe too small — using fallback")
        universe = FALLBACK_UNIVERSE

    log.info(f"🌍 Universe ready: {len(universe)} symbols")
    return universe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    u = build_universe()
    print(f"\nUniverse ({len(u)} symbols):")
    print(", ".join(u[:50]), "...")
