"""
AlgoTrader — Aggressive Multi-Strategy Bot
Dynamic universe: S&P 500 + Most Traded + Most Volatile
Strategies: MA Crossover + RSI + MACD + Bollinger Bands
Fractional shares enabled
"""

import os
import json
import logging
import time
import warnings
from datetime import datetime, timedelta
from typing import List, Optional
import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# Suppress yfinance / pandas warnings
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from universe import build_universe, SP500_STOCKS as FALLBACK_UNIVERSE

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

SHORT_WINDOW       = 8
LONG_WINDOW        = 21
RSI_PERIOD         = 10
RSI_OVERSOLD       = 38
RSI_OVERBOUGHT     = 65
MAX_POSITIONS      = 8
POSITION_SIZE      = 0.20
STOP_LOSS_PCT      = 0.03
TAKE_PROFIT_PCT    = 0.06
MIN_CASH_BUFFER    = 0.05
LOOKBACK_DAYS      = 60
MA_TREND_THRESHOLD = 0.002
SIGNAL_THRESHOLD   = 3
BATCH_SIZE         = 40
BATCH_DELAY        = 2

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


def fetch_bulk(symbols: List[str]) -> dict:
    """Download a batch of symbols in one request. Returns {symbol: df}."""
    try:
        end   = datetime.now()
        start = end - timedelta(days=LOOKBACK_DAYS + 15)
        data  = yf.download(
            symbols,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
        )
        result = {}
        if len(symbols) == 1:
            sym = symbols[0]
            if not data.empty and len(data) >= LONG_WINDOW + 5:
                result[sym] = data
        else:
            for sym in symbols:
                try:
                    if sym in data.columns.get_level_values(0):
                        df = data[sym].dropna(how="all")
                        if not df.empty and len(df) >= LONG_WINDOW + 5:
                            result[sym] = df
                except Exception:
                    pass
        return result
    except Exception as e:
        log.warning(f"Bulk fetch error: {e}")
        return {}


def compute_indicators(df: pd.DataFrame) -> Optional[dict]:
    try:
        close = df["Close"].squeeze()
        vol   = df["Volume"].squeeze()

        if len(close) < LONG_WINDOW + 5:
            return None

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
    except Exception as e:
        log.warning(f"Indicator error: {e}")
        return None


def place_order(api, symbol: str, qty: float, side: str, reason: str = "") -> Optional[dict]:
    try:
        order = api.submit_order(
            symbol=symbol, qty=round(qty, 6), side=side,
            type="market", time_in_force="day",
        )
        log.info(f"✅ {side.upper()} {qty}x {symbol} | {reason} | id={order.id}")
        return {"id": order.id, "symbol": symbol, "qty": qty,
                "side": side, "reason": reason, "time": datetime.now().isoformat()}
    except Exception as e:
        log.error(f"❌ Order failed {side} {symbol}: {e}")
        return None


def calc_shares(portfolio_value: float, price: float, cash: float) -> float:
    alloc = min(portfolio_value * POSITION_SIZE, cash * 0.9)
    if alloc < 1.0:
        return 0.0
    return round(alloc / price, 6)


def check_exits(api, positions: list) -> List[dict]:
    exits = []
    for pos in positions:
        symbol = pos.symbol
        qty    = float(pos.qty)
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
    log.info(f"🤖 AlgoTrader | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
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

    # Build universe
    try:
        universe = build_universe()
    except Exception as e:
        log.warning(f"Universe build failed: {e}")
        universe = FALLBACK_UNIVERSE

    log.info(f"🔍 Scanning {len(universe)} symbols...")

    all_signals    = {}
    all_buys       = []
    all_sells      = []
    batches        = [universe[i:i+BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        log.info(f"📡 Batch {i+1}/{len(batches)}")
        bulk = fetch_bulk(batch)

        for symbol, df in bulk.items():
            ind = compute_indicators(df)
            if ind is None:
                continue
            ind["symbol"] = symbol
            all_signals[symbol] = ind

            if ind["signal"] == "BUY" and symbol not in held:
                all_buys.append(ind)
            elif ind["signal"] in ("SELL", "BEARISH") and symbol in held:
                pos  = held[symbol]
                plpc = float(pos.unrealized_plpc)
                if ind["signal"] == "SELL" or plpc > 0.01 or ind["sell_score"] >= 5:
                    all_sells.append(ind)

        if i < len(batches) - 1:
            time.sleep(BATCH_DELAY)

    log.info(f"📊 Scanned {len(all_signals)} | Buys: {len(all_buys)} | Sells: {len(all_sells)}")

    # Execute sells
    for sig in all_sells:
        symbol = sig["symbol"]
        if symbol not in held:
            continue
        pos   = held[symbol]
        qty   = float(pos.qty)
        order = place_order(api, symbol, qty, "sell", "signal_sell")
        if order:
            state["trades"].append(order)
            state["daily_trades"] += 1
            held.pop(symbol)
            n_positions -= 1

    # Execute buys
    slots    = MAX_POSITIONS - n_positions
    cash     = account["cash"]
    min_cash = account["portfolio_value"] * MIN_CASH_BUFFER

    if slots > 0 and all_buys:
        all_buys.sort(key=lambda x: (x["buy_score"], x.get("vol_ratio", 1)), reverse=True)
        for sig in all_buys[:slots]:
            symbol = sig["symbol"]
            price  = sig["price"]
            if price < 0.50:
                continue
            qty  = calc_shares(account["portfolio_value"], price, cash)
            if qty == 0.0:
                log.info(f"⚠️  Skip {symbol} — insufficient funds")
                continue
            cost = qty * price
            if (cash - cost) < min_cash:
                log.info(f"⚠️  Skip {symbol} — low cash buffer")
                continue
            order = place_order(api, symbol, qty, "buy",
                                f"signal_buy(score={sig['buy_score']})")
            if order:
                order["price"] = price
                state["trades"].append(order)
                state["daily_trades"] += 1
                cash -= cost
                n_positions += 1
    else:
        log.info(f"No buys — slots={slots} candidates={len(all_buys)}")

    state["signals"]  = dict(list(all_signals.items())[:100])
    state["last_run"] = datetime.now().isoformat()
    state["account"]  = account
    save_state(state)

    log.info(f"✅ Done | Scanned={len(all_signals)} | Daily trades={state['daily_trades']}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_bot()
