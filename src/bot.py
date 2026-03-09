"""
AlgoTrader — Aggressive Multi-Strategy Bot
Strategies: MA Crossover (fast) + RSI Mean Reversion + Trend Following
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional
import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

UNIVERSE = [
    "SPY", "QQQ", "IWM", "VTI", "DIA",
    "XLK", "XLV", "XLE", "XLF", "XLY", "XLC", "ARKK",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "BAC", "GS",
    "JNJ", "UNH", "PFE",
    "COST", "WMT", "HD",
    "XOM", "CVX",
    "AMD", "COIN", "PLTR", "SOFI", "UBER", "ABNB", "SHOP",
    "TQQQ", "SOXL",
]

SHORT_WINDOW       = 8
LONG_WINDOW        = 21
RSI_PERIOD         = 10
RSI_OVERSOLD       = 38
RSI_OVERBOUGHT     = 65
MAX_POSITIONS      = 8
POSITION_SIZE      = 0.11
STOP_LOSS_PCT      = 0.03
TAKE_PROFIT_PCT    = 0.06
MIN_CASH_BUFFER    = 0.08
LOOKBACK_DAYS      = 60
MA_TREND_THRESHOLD = 0.002
SIGNAL_THRESHOLD   = 3

STATE_FILE = Path("logs/state.json")


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"trades": [], "signals": {}, "last_run": None, "daily_trades": 0, "trade_date": None}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_api():
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")


def get_account(api) -> dict:
    acc = api.get_account()
    return {
        "portfolio_value": float(acc.portfolio_value),
        "cash":            float(acc.cash),
        "buying_power":    float(acc.buying_power),
        "equity":          float(acc.equity),
        "status":          acc.status,
    }


def fetch_data(symbol: str) -> Optional[pd.DataFrame]:
    try:
        end   = datetime.now()
        start = end - timedelta(days=LOOKBACK_DAYS + 15)
        df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df.empty or len(df) < LONG_WINDOW + 5:
            return None
        return df
    except Exception as e:
        log.warning(f"Data fetch failed {symbol}: {e}")
        return None


def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"].squeeze()
    vol   = df["Volume"].squeeze()

    sma_short = close.rolling(SHORT_WINDOW).mean()
    sma_long  = close.rolling(LONG_WINDOW).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    ema12  = close.ewm(span=12).mean()
    ema26  = close.ewm(span=26).mean()
    macd   = ema12 - ema26
    macd_s = macd.ewm(span=9).mean()
    hist   = macd - macd_s

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std

    avg_vol   = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / avg_vol.iloc[-1]) if float(avg_vol.iloc[-1]) > 0 else 1.0

    price     = float(close.iloc[-1])
    ss        = float(sma_short.iloc[-1])
    sl        = float(sma_long.iloc[-1])
    ss_prev   = float(sma_short.iloc[-2])
    sl_prev   = float(sma_long.iloc[-2])
    rsi_val   = float(rsi.iloc[-1])
    macd_val  = float(macd.iloc[-1])
    macd_sval = float(macd_s.iloc[-1])
    hist_val  = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2])
    bb_low_v  = float(bb_low.iloc[-1])
    bb_up_v   = float(bb_up.iloc[-1])
    bb_mid_v  = float(bb_mid.iloc[-1])

    buy_score = sell_score = 0

    if (ss_prev <= sl_prev) and (ss > sl): buy_score  += 3
    if (ss_prev >= sl_prev) and (ss < sl): sell_score += 3

    ma_gap = (ss - sl) / sl if sl > 0 else 0
    if ma_gap >  MA_TREND_THRESHOLD: buy_score  += 1
    if ma_gap < -MA_TREND_THRESHOLD: sell_score += 1

    if rsi_val < RSI_OVERSOLD:   buy_score  += 2
    if rsi_val > RSI_OVERBOUGHT: sell_score += 2

    if hist_val > 0 and hist_prev <= 0: buy_score  += 2
    if hist_val < 0 and hist_prev >= 0: sell_score += 2
    if macd_val > macd_sval:            buy_score  += 1
    if macd_val < macd_sval:            sell_score += 1

    if price <= bb_low_v: buy_score  += 2
    if price >= bb_up_v:  sell_score += 2

    if vol_ratio > 1.5:
        buy_score  = int(buy_score  * 1.2)
        sell_score = int(sell_score * 1.2)

    if buy_score >= SIGNAL_THRESHOLD:
        signal = "BUY"
    elif sell_score >= SIGNAL_THRESHOLD:
        signal = "SELL"
    elif buy_score > sell_score:
        signal = "BULLISH"
    elif sell_score > buy_score:
        signal = "BEARISH"
    else:
        signal = "HOLD"

    return {
        "price": price, "sma_short": ss, "sma_long": sl,
        "rsi": rsi_val, "macd": macd_val, "macd_signal": macd_sval,
        "macd_hist": hist_val, "bb_low": bb_low_v, "bb_up": bb_up_v,
        "bb_mid": bb_mid_v, "vol_ratio": vol_ratio,
        "buy_score": buy_score, "sell_score": sell_score,
        "signal": signal, "above_trend": ss > sl,
        "timestamp": datetime.now().isoformat(),
    }


def place_order(api, symbol: str, qty: int, side: str, reason: str = "") -> Optional[dict]:
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side,
                                 type="market", time_in_force="day")
        log.info(f"✅ {side.upper()} {qty}x {symbol} | {reason} | id={order.id}")
        return {"id": order.id, "symbol": symbol, "qty": qty,
                "side": side, "reason": reason, "time": datetime.now().isoformat()}
    except Exception as e:
        log.error(f"❌ Order failed {side} {symbol}: {e}")
        return None


def calc_shares(portfolio_value: float, price: float, cash: float) -> int:
    """Calculate affordable whole shares, never exceeding available cash."""
    alloc  = min(portfolio_value * POSITION_SIZE, cash * 0.9)
    shares = int(alloc / price)
    return max(0, shares)


def check_exits(api, positions: list) -> List[dict]:
    exits = []
    for pos in positions:
        symbol = pos.symbol
        qty    = int(float(pos.qty))
        plpc   = float(pos.unrealized_plpc)
        if plpc <= -STOP_LOSS_PCT:
            log.warning(f"🛑 STOP LOSS {symbol}: {plpc:.1%}")
            order = place_order(api, symbol, qty, "sell", "stop_loss")
            if order:
                exits.append(order)
        elif plpc >= TAKE_PROFIT_PCT:
            log.info(f"🎯 TAKE PROFIT {symbol}: {plpc:.1%}")
            order = place_order(api, symbol, qty, "sell", "take_profit")
            if order:
                exits.append(order)
    return exits


def run_bot():
    log.info("=" * 60)
    log.info(f"🤖 AggressiveBot | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    log.info("=" * 60)

    api   = get_api()
    state = load_state()

    account = get_account(api)
    log.info(f"💰 Portfolio: ${account['portfolio_value']:,.2f} | Cash: ${account['cash']:,.2f}")

    if account["status"] != "ACTIVE":
        log.error("Account not active — aborting")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("trade_date") != today:
        state["daily_trades"] = 0
        state["trade_date"]   = today

    positions   = api.list_positions()
    held        = {p.symbol: p for p in positions}
    n_positions = len(positions)

    exits = check_exits(api, positions)
    if exits:
        state["trades"].extend(exits)
        state["daily_trades"] += len(exits)
        positions   = api.list_positions()
        held        = {p.symbol: p for p in positions}
        n_positions = len(positions)
        time.sleep(1)

    all_signals     = {}
    buy_candidates  = []
    sell_candidates = []

    for symbol in UNIVERSE:
        df = fetch_data(symbol)
        if df is None:
            continue
        try:
            ind = compute_indicators(df)
            ind["symbol"] = symbol
            all_signals[symbol] = ind
            if ind["signal"] == "BUY" and symbol not in held:
                buy_candidates.append(ind)
            elif ind["signal"] in ("SELL", "BEARISH") and symbol in held:
                pos  = held[symbol]
                plpc = float(pos.unrealized_plpc)
                if ind["signal"] == "SELL" or plpc > 0.01 or ind["sell_score"] >= 5:
                    sell_candidates.append(ind)
        except Exception as e:
            log.warning(f"Signal error {symbol}: {e}")
        time.sleep(0.1)

    for sig in sell_candidates:
        symbol = sig["symbol"]
        if symbol not in held:
            continue
        pos   = held[symbol]
        qty   = int(float(pos.qty))
        order = place_order(api, symbol, qty, "sell", "signal_sell")
        if order:
            state["trades"].append(order)
            state["daily_trades"] += 1
            held.pop(symbol)
            n_positions -= 1

    slots    = MAX_POSITIONS - n_positions
    cash     = account["cash"]
    min_cash = account["portfolio_value"] * MIN_CASH_BUFFER

    if slots > 0 and buy_candidates:
        buy_candidates.sort(key=lambda x: x["buy_score"], reverse=True)
        for sig in buy_candidates[:slots]:
            symbol = sig["symbol"]
            price  = sig["price"]
            if price < 1.0:
                continue
            qty  = calc_shares(account["portfolio_value"], price, cash)
            if qty == 0:
                log.info(f"⚠️  Skip {symbol} — can't afford 1 share at ${price:.2f}")
                continue
            cost = qty * price
            if (cash - cost) < min_cash:
                log.info(f"⚠️  Skip {symbol} — low cash")
                continue
            order = place_order(api, symbol, qty, "buy", f"signal_buy(score={sig['buy_score']})")
            if order:
                order["price"] = price
                state["trades"].append(order)
                state["daily_trades"] += 1
                cash -= cost
                n_positions += 1
    else:
        log.info(f"No buys — slots={slots} candidates={len(buy_candidates)}")

    state["signals"]  = all_signals
    state["last_run"] = datetime.now().isoformat()
    state["account"]  = account
    save_state(state)

    log.info(f"✅ Done | Scanned={len(all_signals)} | Daily trades={state['daily_trades']}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_bot()
