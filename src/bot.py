"""
Trading Bot - Moving Average Crossover Strategy
Trades a mix of Stocks and ETFs via Alpaca API
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Universe: diversified mix of liquid stocks + ETFs
UNIVERSE = [
    # Broad market ETFs
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "VTI",   # Total US Market
    # Sector ETFs
    "XLK",   # Technology
    "XLV",   # Healthcare
    "XLE",   # Energy
    # Individual stocks (high liquidity)
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "NVDA",  # Nvidia
    "JPM",   # JPMorgan
]

# Strategy Parameters
SHORT_WINDOW    = 20   # Fast MA (days)
LONG_WINDOW     = 50   # Slow MA (days)
MAX_POSITIONS   = 5    # Max concurrent holdings
POSITION_SIZE   = 0.18 # 18% of portfolio per position (leaving buffer)
STOP_LOSS_PCT   = 0.05 # 5% stop loss
TAKE_PROFIT_PCT = 0.12 # 12% take profit
MIN_CASH_BUFFER = 0.10 # Keep 10% cash minimum
LOOKBACK_DAYS   = 80   # Days of historical data to fetch

STATE_FILE = Path("logs/state.json")


# ── State Management ──────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"trades": [], "signals": {}, "last_run": None}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Alpaca Connection ─────────────────────────────────────────────────────────
def get_api() -> tradeapi.REST:
    return tradeapi.REST(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        ALPACA_BASE_URL,
        api_version="v2",
    )


def get_account_info(api: tradeapi.REST) -> dict:
    acc = api.get_account()
    return {
        "portfolio_value": float(acc.portfolio_value),
        "cash":            float(acc.cash),
        "buying_power":    float(acc.buying_power),
        "equity":          float(acc.equity),
        "status":          acc.status,
    }


# ── Market Data ───────────────────────────────────────────────────────────────
def fetch_historical(symbol: str, days: int = LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from Yahoo Finance (free, no API key needed)."""
    try:
        end   = datetime.now()
        start = end - timedelta(days=days + 10)
        df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df.empty or len(df) < LONG_WINDOW:
            return None
        df = df.tail(days)
        return df
    except Exception as e:
        log.warning(f"Failed to fetch {symbol}: {e}")
        return None


def compute_signals(df: pd.DataFrame) -> dict:
    """Compute Moving Average Crossover signals + RSI filter."""
    close = df["Close"].squeeze()

    sma_short = close.rolling(SHORT_WINDOW).mean()
    sma_long  = close.rolling(LONG_WINDOW).mean()

    # RSI (14) as confirmation filter
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    latest_short = float(sma_short.iloc[-1])
    latest_long  = float(sma_long.iloc[-1])
    prev_short   = float(sma_short.iloc[-2])
    prev_long    = float(sma_long.iloc[-2])
    latest_rsi   = float(rsi.iloc[-1])
    latest_price = float(close.iloc[-1])

    # Golden cross: short crosses above long → BUY
    # Death cross:  short crosses below long → SELL
    bullish_cross = (prev_short <= prev_long) and (latest_short > latest_long)
    bearish_cross = (prev_short >= prev_long) and (latest_short < latest_long)

    # RSI filter: don't buy if overbought (>70), don't sell if oversold (<30)
    rsi_ok_buy  = latest_rsi < 70
    rsi_ok_sell = latest_rsi > 30

    signal = "HOLD"
    if bullish_cross and rsi_ok_buy:
        signal = "BUY"
    elif bearish_cross and rsi_ok_sell:
        signal = "SELL"
    elif latest_short > latest_long:
        signal = "BULLISH"   # In uptrend but no fresh cross
    else:
        signal = "BEARISH"

    return {
        "symbol":      df.index.name or "N/A",
        "price":       latest_price,
        "sma_short":   latest_short,
        "sma_long":    latest_long,
        "rsi":         latest_rsi,
        "signal":      signal,
        "above_trend": latest_short > latest_long,
        "timestamp":   datetime.now().isoformat(),
    }


# ── Position Sizing ───────────────────────────────────────────────────────────
def calc_shares(portfolio_value: float, price: float, cash: float) -> int:
    alloc = min(portfolio_value * POSITION_SIZE, cash * 0.9)
    shares = int(alloc / price)
    return max(0, shares)


# ── Order Execution ───────────────────────────────────────────────────────────
def place_order(api: tradeapi.REST, symbol: str, qty: int, side: str) -> Optional[dict]:
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
        )
        log.info(f"✅ ORDER {side.upper()} {qty}x {symbol} — id={order.id}")
        return {"id": order.id, "symbol": symbol, "qty": qty,
                "side": side, "time": datetime.now().isoformat()}
    except Exception as e:
        log.error(f"❌ Order failed {side} {symbol}: {e}")
        return None


def check_stop_loss_take_profit(api: tradeapi.REST, positions: list, state: dict) -> List[dict]:
    """Exit positions that hit stop-loss or take-profit thresholds."""
    exits = []
    for pos in positions:
        symbol      = pos.symbol
        qty         = int(float(pos.qty))
        unrealised  = float(pos.unrealized_plpc)  # e.g. 0.05 = 5%

        if unrealised <= -STOP_LOSS_PCT:
            log.warning(f"🛑 STOP LOSS hit for {symbol}: {unrealised:.1%}")
            order = place_order(api, symbol, qty, "sell")
            if order:
                order["reason"] = "stop_loss"
                exits.append(order)

        elif unrealised >= TAKE_PROFIT_PCT:
            log.info(f"🎯 TAKE PROFIT hit for {symbol}: {unrealised:.1%}")
            order = place_order(api, symbol, qty, "sell")
            if order:
                order["reason"] = "take_profit"
                exits.append(order)

    return exits


# ── Main Bot Logic ────────────────────────────────────────────────────────────
def run_bot():
    log.info("=" * 60)
    log.info("🤖 Trading Bot Starting")
    log.info("=" * 60)

    api   = get_api()
    state = load_state()

    # Account check
    account = get_account_info(api)
    log.info(f"💰 Portfolio: £{account['portfolio_value']:,.2f} | "
             f"Cash: £{account['cash']:,.2f} | Status: {account['status']}")

    if account["status"] != "ACTIVE":
        log.error("Account not active — aborting.")
        return

    # Get current positions
    try:
        positions     = api.list_positions()
        held_symbols  = {p.symbol for p in positions}
        n_positions   = len(positions)
    except Exception as e:
        log.error(f"Could not fetch positions: {e}")
        return

    log.info(f"📊 Current positions ({n_positions}): {held_symbols or 'none'}")

    # ── Step 1: Check stop-loss / take-profit ─────────────────────────────────
    exits = check_stop_loss_take_profit(api, positions, state)
    if exits:
        state["trades"].extend(exits)
        # Refresh positions after exits
        positions    = api.list_positions()
        held_symbols = {p.symbol for p in positions}
        n_positions  = len(positions)
        time.sleep(2)

    # ── Step 2: Scan universe for signals ─────────────────────────────────────
    all_signals = {}
    buy_candidates = []

    for symbol in UNIVERSE:
        df = fetch_historical(symbol)
        if df is None:
            continue
        df.index.name = symbol
        sig = compute_signals(df)
        sig["symbol"] = symbol
        all_signals[symbol] = sig

        if sig["signal"] == "BUY" and symbol not in held_symbols:
            buy_candidates.append(sig)
            log.info(f"📈 BUY signal: {symbol} | price={sig['price']:.2f} | "
                     f"SMA{SHORT_WINDOW}={sig['sma_short']:.2f} | "
                     f"SMA{LONG_WINDOW}={sig['sma_long']:.2f} | RSI={sig['rsi']:.1f}")

        elif sig["signal"] == "SELL" and symbol in held_symbols:
            # Exit signal for held position
            pos = next((p for p in positions if p.symbol == symbol), None)
            if pos:
                qty   = int(float(pos.qty))
                order = place_order(api, symbol, qty, "sell")
                if order:
                    order["reason"] = "signal_sell"
                    state["trades"].append(order)
                    held_symbols.discard(symbol)
                    n_positions -= 1
                    log.info(f"📉 SELL signal executed: {symbol}")

    # ── Step 3: Execute buy signals ───────────────────────────────────────────
    slots_available = MAX_POSITIONS - n_positions
    cash_available  = account["cash"]
    min_cash        = account["portfolio_value"] * MIN_CASH_BUFFER

    if slots_available > 0 and buy_candidates:
        # Sort by how far short MA is above long MA (strongest trend first)
        buy_candidates.sort(
            key=lambda x: (x["sma_short"] - x["sma_long"]) / x["sma_long"],
            reverse=True,
        )

        for sig in buy_candidates[:slots_available]:
            symbol = sig["symbol"]
            price  = sig["price"]
            qty = calc_shares(account["portfolio_value"], price, cash)
            cost   = qty * price

            if (cash_available - cost) < min_cash:
                log.info(f"⚠️  Skipping {symbol} — insufficient cash buffer")
                continue

            order = place_order(api, symbol, qty, "buy")
            if order:
                order["reason"]  = "signal_buy"
                order["price"]   = price
                state["trades"].append(order)
                cash_available  -= cost
                n_positions     += 1
    else:
        if slots_available == 0:
            log.info("📊 Max positions reached — monitoring only")
        elif not buy_candidates:
            log.info("🔍 No BUY signals found this cycle")

    # ── Step 4: Save state & summary ─────────────────────────────────────────
    state["signals"]  = all_signals
    state["last_run"] = datetime.now().isoformat()
    state["account"]  = account
    save_state(state)

    log.info(f"✅ Bot cycle complete | Signals scanned: {len(all_signals)} | "
             f"Trades this run: {len(exits) + max(0, n_positions - (MAX_POSITIONS - slots_available))}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_bot()
