"""
Dynamic Universe Builder
Combines: S&P 500 + Most Traded + Most Volatile
Uses multiple fallback methods for reliability
"""

import logging
import time
import requests
import pandas as pd
from typing import List

log = logging.getLogger(__name__)

# ── Hardcoded S&P 500 subset + extras as reliable fallback ───────────────────
SP500_STOCKS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "JPM","TSLA","UNH","V","XOM","MA","JNJ","PG","COST","HD","MRK","ABBV",
    "CVX","CRM","BAC","NFLX","AMD","PEP","KO","TMO","ORCL","ACN","LIN","MCD",
    "CSCO","ABT","GE","DHR","TXN","CAT","AMGN","ISRG","NOW","INTU","UBER",
    "SPGI","PFE","AXP","RTX","QCOM","HON","MS","GS","BLK","T","NEE","LOW",
    "UNP","BMY","DE","SCHW","VRTX","AMAT","MDT","SYK","ELV","ADI","GILD",
    "TJX","REGN","CB","PLD","MMC","ADP","LRCX","C","MU","ETN","ZTS","BSX",
    "SO","DUK","WM","CME","BDX","PH","ITW","NOC","EMR","FCX","CEG","SLB",
    "CL","NSC","APH","AON","MCO","FDX","HUM","TGT","USB","ICE","ECL","GD",
    "PSA","WMB","HCA","ORLY","CVS","AIG","F","GM","PYPL","SQ","COIN","PLTR",
    "SOFI","HOOD","RBLX","SNAP","PINS","LYFT","DASH","ABNB","SHOP","SPOT",
    "ZM","DOCU","TWLO","NET","DDOG","SNOW","PATH","U","GTLB","MDB","CRWD",
    "ZS","OKTA","HUBS","TEAM","WDAY","VEEV","PANW","FTNT","SPLK","COUP",
    "SPY","QQQ","IWM","VTI","DIA","XLK","XLV","XLE","XLF","XLY","XLC",
    "ARKK","TQQQ","SOXL","SQQQ","GLD","SLV","TLT","HYG","EEM","EFA",
]

def get_sp500() -> List[str]:
    """Fetch S&P 500 tickers — tries multiple sources with fallback."""

    # Method 1: slickcharts.com
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }
        r = requests.get("https://slickcharts.com/sp500", headers=headers, timeout=10)
        df = pd.read_html(r.text)[0]
        tickers = df["Symbol"].tolist()
        tickers = [str(t).replace(".", "-") for t in tickers if str(t) != "nan"]
        if len(tickers) > 400:
            log.info(f"✅ S&P 500 (slickcharts): {len(tickers)} tickers")
            return tickers
    except Exception as e:
        log.warning(f"slickcharts failed: {e}")

    # Method 2: Wikipedia with custom headers
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; research bot)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=10
        )
        df = pd.read_html(r.text)[0]
        tickers = df["Symbol"].tolist()
        tickers = [str(t).replace(".", "-") for t in tickers]
        if len(tickers) > 400:
            log.info(f"✅ S&P 500 (wikipedia): {len(tickers)} tickers")
            return tickers
    except Exception as e:
        log.warning(f"Wikipedia S&P 500 failed: {e}")

    # Method 3: Use hardcoded list
    log.info(f"✅ S&P 500 (hardcoded): {len(SP500_STOCKS)} tickers")
    return SP500_STOCKS


def get_most_traded() -> List[str]:
    """Get top 100 most actively traded stocks via Yahoo Finance screener."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {"scrIds": "most_actives", "count": 100, "lang": "en-US", "region": "US"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        quotes = data["finance"]["result"][0]["quotes"]
        tickers = [q["symbol"] for q in quotes if "." not in q["symbol"]]
        log.info(f"✅ Most traded: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(f"Most traded fetch failed: {e}")
        return []


def get_most_volatile() -> List[str]:
    """Get top gainers and losers today."""
    tickers = []
    try:
        for scrId in ["day_gainers", "day_losers"]:
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            params = {"scrIds": scrId, "count": 50, "lang": "en-US", "region": "US"}
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r = requests.get(url, params=params, headers=headers, timeout=10)
            data = r.json()
            quotes = data["finance"]["result"][0]["quotes"]
            batch = [q["symbol"] for q in quotes if "." not in q["symbol"]]
            tickers.extend(batch)
            time.sleep(0.5)
        log.info(f"✅ Most volatile: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(f"Most volatile fetch failed: {e}")
        return []


def filter_tradeable(tickers: List[str], max_symbols: int = 300) -> List[str]:
    """Filter out untradeable symbols and deduplicate."""
    seen = set()
    clean = []
    bad_suffixes = {"W", "WS", "U", "R", "P", "Q"}

    for t in tickers:
        t = t.strip().upper()
        if not t or t in seen:
            continue
        if any(t.endswith(s) for s in bad_suffixes):
            continue
        if len(t) > 5:
            continue
        seen.add(t)
        clean.append(t)
        if len(clean) >= max_symbols:
            break

    return clean


def build_universe() -> List[str]:
    """Build combined smart universe from all three sources."""
    log.info("🌍 Building dynamic universe...")

    all_tickers = []

    volatile = get_most_volatile()
    all_tickers.extend(volatile)
    time.sleep(0.5)

    traded = get_most_traded()
    all_tickers.extend(traded)
    time.sleep(0.5)

    sp500 = get_sp500()
    all_tickers.extend(sp500)

    universe = filter_tradeable(all_tickers, max_symbols=300)

    if len(universe) < 20:
        log.warning("Dynamic universe too small — using fallback")
        universe = filter_tradeable(SP500_STOCKS)

    log.info(f"🌍 Universe ready: {len(universe)} symbols")
    return universe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    u = build_universe()
    print(f"\nUniverse ({len(u)} symbols):")
    print(", ".join(u[:50]), "...")
