"""
Dynamic Universe Builder - clean version with no noisy output
"""

import logging
import time
import requests
from typing import List

log = logging.getLogger(__name__)

SP500_STOCKS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK.B","LLY","AVGO","JPM",
    "TSLA","UNH","V","XOM","MA","JNJ","PG","COST","HD","MRK","ABBV","CVX",
    "CRM","BAC","NFLX","AMD","PEP","KO","TMO","ORCL","ACN","MCD","CSCO",
    "ABT","GE","DHR","TXN","CAT","AMGN","ISRG","NOW","INTU","UBER","SPGI",
    "PFE","AXP","RTX","QCOM","HON","MS","GS","BLK","T","NEE","LOW","UNP",
    "BMY","DE","SCHW","VRTX","AMAT","MDT","SYK","ELV","ADI","GILD","TJX",
    "REGN","CB","PLD","MMC","ADP","LRCX","C","MU","ETN","ZTS","BSX","SO",
    "DUK","WM","CME","BDX","PH","ITW","NOC","EMR","FCX","SLB","CL","NSC",
    "APH","AON","MCO","FDX","HUM","TGT","USB","ICE","ECL","GD","PSA","WMB",
    "HCA","ORLY","CVS","AIG","F","GM","PYPL","COIN","PLTR","SOFI","HOOD",
    "RBLX","SNAP","PINS","LYFT","DASH","ABNB","SHOP","SPOT","ZM","NET",
    "DDOG","SNOW","CRWD","ZS","PANW","FTNT","MDB","WDAY","VEEV","HUBS",
    "SPY","QQQ","IWM","VTI","DIA","XLK","XLV","XLE","XLF","XLY","XLC",
    "ARKK","TQQQ","SOXL","GLD","TLT","EEM","MU","INTC","NFLX","DIS","BA",
]


def get_most_traded() -> List[str]:
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {"scrIds": "most_actives", "count": 50, "lang": "en-US", "region": "US"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        data = r.json()
        quotes = data["finance"]["result"][0]["quotes"]
        tickers = [q["symbol"] for q in quotes if "." not in q["symbol"]]
        log.info(f"✅ Most traded: {len(tickers)} tickers")
        return tickers
    except Exception:
        return []


def get_most_volatile() -> List[str]:
    tickers = []
    try:
        for scrId in ["day_gainers", "day_losers"]:
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            params = {"scrIds": scrId, "count": 30, "lang": "en-US", "region": "US"}
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, params=params, headers=headers, timeout=8)
            data = r.json()
            quotes = data["finance"]["result"][0]["quotes"]
            tickers.extend([q["symbol"] for q in quotes if "." not in q["symbol"]])
            time.sleep(0.3)
        log.info(f"✅ Most volatile: {len(tickers)} tickers")
    except Exception:
        pass
    return tickers


def filter_tradeable(tickers: List[str], max_symbols: int = 150) -> List[str]:
    seen  = set()
    clean = []
    for t in tickers:
        t = t.strip().upper()
        if not t or t in seen or len(t) > 5:
            continue
        if t[-1] in {"W", "R", "P", "Q"}:
            continue
        seen.add(t)
        clean.append(t)
        if len(clean) >= max_symbols:
            break
    return clean


def build_universe() -> List[str]:
    log.info("🌍 Building universe...")
    all_tickers = []
    all_tickers.extend(get_most_volatile())
    time.sleep(0.3)
    all_tickers.extend(get_most_traded())
    all_tickers.extend(SP500_STOCKS)
    universe = filter_tradeable(all_tickers, max_symbols=150)
    if len(universe) < 20:
        universe = filter_tradeable(SP500_STOCKS)
    log.info(f"🌍 Universe: {len(universe)} symbols")
    return universe
