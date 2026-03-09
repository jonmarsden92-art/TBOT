"""
Generate a performance report from bot state and Alpaca account.
Saves report.json used by the dashboard.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

STATE_FILE  = Path("logs/state.json")
REPORT_FILE = Path("logs/report.json")


def get_api() -> Optional[tradeapi.REST]:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    try:
        return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    except Exception as e:
        log.error(f"API connection failed: {e}")
        return None


def build_report() -> dict:
    state = {}
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)

    report = {
        "generated_at":   datetime.now().isoformat(),
        "account":        state.get("account", {}),
        "last_run":       state.get("last_run"),
        "trades":         state.get("trades", []),
        "signals":        state.get("signals", {}),
        "positions":      [],
        "performance":    {},
        "equity_history": [],
    }

    api = get_api()
    if api:
        try:
            acc = api.get_account()
            report["account"] = {
                "portfolio_value": float(acc.portfolio_value),
                "cash":            float(acc.cash),
                "buying_power":    float(acc.buying_power),
                "equity":          float(acc.equity),
                "status":          acc.status,
            }

            # Current positions with live P&L
            positions = api.list_positions()
            report["positions"] = [
                {
                    "symbol":           p.symbol,
                    "qty":              float(p.qty),
                    "avg_entry":        float(p.avg_entry_price),
                    "current_price":    float(p.current_price),
                    "market_value":     float(p.market_value),
                    "unrealized_pl":    float(p.unrealized_pl),
                    "unrealized_plpc":  float(p.unrealized_plpc),
                    "side":             p.side,
                }
                for p in positions
            ]

            # Portfolio history (last 30 days)
            try:
                hist = api.get_portfolio_history(period="1M", timeframe="1D")
                if hist and hist.equity:
                    report["equity_history"] = [
                        {"timestamp": ts, "equity": eq}
                        for ts, eq in zip(hist.timestamp, hist.equity)
                        if eq is not None
                    ]
            except Exception:
                pass

            # Performance summary
            trades = report["trades"]
            buys   = [t for t in trades if t.get("side") == "buy"]
            sells  = [t for t in trades if t.get("side") == "sell"]

            report["performance"] = {
                "total_trades":    len(trades),
                "total_buys":      len(buys),
                "total_sells":     len(sells),
                "open_positions":  len(positions),
                "total_pnl":       sum(p["unrealized_pl"] for p in report["positions"]),
                "start_value":     50.0,   # Initial capital
                "current_value":   float(acc.portfolio_value),
                "return_pct":      ((float(acc.portfolio_value) - 50.0) / 50.0) * 100,
            }

        except Exception as e:
            log.error(f"Failed to build live report: {e}")

    REPORT_FILE.parent.mkdir(exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"📊 Report saved to {REPORT_FILE}")
    return report


if __name__ == "__main__":
    report = build_report()
    acc    = report.get("account", {})
    perf   = report.get("performance", {})
    print(f"\n{'='*50}")
    print(f"  Portfolio: ${acc.get('portfolio_value', 0):,.2f}")
    print(f"  Cash:      ${acc.get('cash', 0):,.2f}")
    print(f"  Return:    {perf.get('return_pct', 0):+.2f}%")
    print(f"  Trades:    {perf.get('total_trades', 0)}")
    print(f"  Positions: {perf.get('open_positions', 0)}")
    print(f"{'='*50}\n")
