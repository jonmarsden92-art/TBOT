"""
AlgoTrader v2 — Upgraded Bot
Improvements:
  - Alpaca market data API (no yfinance for price data)
  - Kelly Criterion position sizing
  - Trailing stop losses
  - Market regime filter (SPY trend)
  - Smart universe: full scan at open, quick scan after
  - All warnings suppressed, clean logs
"""

import os
import json
import logging
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

from universe import build_universe, SP500_STOCKS as FALLBACK_UNIVERSE

# ── Logging ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)
for noisy in ("urllib3", "requests", "alpaca_trade_api", "yfinance", "peewee"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# ── Config ────────────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

SHORT_WINDOW        = 8
LONG_WINDOW         = 21
RSI_PERIOD          = 10
RSI_OVERSOLD        = 38
RSI_OVERBOUGHT      = 65
MAX_POSITIONS       = 8
MIN_POSITION_SIZE   = 0.10   # minimum 10% per trade
MAX_POSITION_SIZE   = 0.25   # maximum 25% per trade (Kelly cap)
TRAILING_STOP_PCT   = 0.04   # 4% trailing stop from peak
STOP_LOSS_PCT       = 0.05   # 5% hard stop loss
TAKE_PROFIT_PCT     = 0.08   # 8% take profit
MIN_CASH_BUFFER     = 0.05
LOOKBACK_DAYS       = 60
MA_TREND_THRESHOLD  = 0.002
SIGNAL_THRESHOLD    = 3
BATCH_SIZE          = 40
BATCH_DELAY         = 1
MARKET_OPEN_HOUR    = 14   # UTC (9:30 AM ET)
MARKET_OPEN_MINUTE  = 30
FULL_SCAN_WINDOW    = 45   # minutes after open to do full scan

STATE_FILE = Path("logs/state.json")


# ── State ─────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "trades": [], "signals": {}, "last_run": None,
        "daily_trades": 0, "trade_date": None,
        "peak_prices": {},   # for trailing stops: {symbol: peak_price}
        "universe_cache": [], "universe_date": None,
    }


def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Alpaca ────────────────────────────────────────────────────────────────────
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


# ── Market Data (Alpaca) ──────────────────────────────────────────────────────
def fetch_bars_alpaca(api, symbols: List[str], days: int = 75) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV bars using Alpaca's market data API."""
    result = {}
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # Alpaca allows up to 200 symbols per request
    for i in range(0, len(symbols), 200):
        batch = symbols[i:i+200]
        try:
            bars = api.get_bars(
                batch,
                tradeapi.rest.TimeFrame.Day,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                adjustment="all",
                feed="iex",
            ).df

            if bars.empty:
                continue

            # Multi-symbol response has a 'symbol' column
            if "symbol" in bars.columns:
                for sym in batch:
                    sym_bars = bars[bars["symbol"] == sym].copy()
                    if len(sym_bars) >= LONG_WINDOW + 5:
                        sym_bars.index = pd.to_datetime(sym_bars.index)
                        result[sym] = sym_bars
            else:
                # Single symbol
                if len(bars) >= LONG_WINDOW + 5:
                    result[batch[0]] = bars

        except Exception as e:
            log.warning(f"Alpaca bars error (batch {i}): {e}")

    return result


def get_spy_trend(api) -> bool:
    """Return True if SPY is in an uptrend (above 50-day MA) — market regime filter."""
    try:
        bars = fetch_bars_alpaca(api, ["SPY"], days=80)
        if "SPY" not in bars:
            return True  # assume bullish if data unavailable
        close = bars["SPY"]["close"]
        ma50  = close.rolling(50).mean()
        trend = float(close.iloc[-1]) > float(ma50.iloc[-1])
        regime = "📈 BULL" if trend else "📉 BEAR"
        log.info(f"🌍 Market regime: {regime} (SPY vs 50MA)")
        return trend
    except Exception:
        return True  # default to allowing trades


# ── Indicators ────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> Optional[dict]:
    try:
        # Alpaca uses lowercase column names
        close_col = "close" if "close" in df.columns else "Close"
        vol_col   = "volume" if "volume" in df.columns else "Volume"

        close = df[close_col].squeeze().astype(float)
        vol   = df[vol_col].squeeze().astype(float)

        if len(close) < LONG_WINDOW + 5:
            return None

        sma_short = close.rolling(SHORT_WINDOW).mean()
        sma_long  = close.rolling(LONG_WINDOW).mean()

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))

        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        macd_s = macd.ewm(span=9, adjust=False).mean()
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

        # MA crossover
        if (ss_prev <= sl_prev) and (ss > sl): buy_score  += 3
        if (ss_prev >= sl_prev) and (ss < sl): sell_score += 3

        # MA gap trend
        ma_gap = (ss - sl) / sl if sl > 0 else 0
        if ma_gap >  MA_TREND_THRESHOLD: buy_score  += 1
        if ma_gap < -MA_TREND_THRESHOLD: sell_score += 1

        # RSI
        if rsi_val < RSI_OVERSOLD:   buy_score  += 2
        if rsi_val > RSI_OVERBOUGHT: sell_score += 2

        # MACD
        if hist_val > 0 and hist_prev <= 0: buy_score  += 2
        if hist_val < 0 and hist_prev >= 0: sell_score += 2
        if macd_val > macd_sval:            buy_score  += 1
        if macd_val < macd_sval:            sell_score += 1

        # Bollinger Bands
        if price <= bb_low_v: buy_score  += 2
        if price >= bb_up_v:  sell_score += 2

        # Volume boost
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

        # Win rate estimate for Kelly (based on score strength)
        win_rate  = min(0.75, 0.45 + (buy_score * 0.04))
        lose_rate = 1 - win_rate
        avg_win   = TAKE_PROFIT_PCT
        avg_loss  = STOP_LOSS_PCT
        kelly     = (win_rate / avg_loss) - (lose_rate / avg_win) if avg_win > 0 else 0
        kelly     = max(0.0, min(kelly, 1.0))  # clamp 0-1

        return {
            "price": price, "sma_short": ss, "sma_long": sl,
            "rsi": rsi_val, "macd": macd_val, "macd_signal": macd_sval,
            "macd_hist": hist_val, "bb_low": bb_low_v, "bb_up": bb_up_v,
            "bb_mid": bb_mid_v, "vol_ratio": vol_ratio,
            "buy_score": buy_score, "sell_score": sell_score,
            "signal": signal, "above_trend": ss > sl,
            "kelly": kelly,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        log.debug(f"Indicator error: {e}")
        return None


# ── Orders ────────────────────────────────────────────────────────────────────
def place_order(api, symbol: str, qty: float, side: str, reason: str = "") -> Optional[dict]:
    try:
        order = api.submit_order(
            symbol=symbol, qty=round(qty, 6), side=side,
            type="market", time_in_force="day",
        )
        log.info(f"✅ {side.upper()} {round(qty,6)}x {symbol} | {reason} | id={order.id}")
        return {
            "id": order.id, "symbol": symbol, "qty": qty,
            "side": side, "reason": reason,
            "time": datetime.now().isoformat(),
        }
    except Exception as e:
        log.error(f"❌ Order failed {side} {symbol}: {e}")
        return None


def kelly_position_size(portfolio_value: float, cash: float, kelly: float) -> float:
    """Kelly Criterion position sizing, capped between min and max."""
    fraction  = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * kelly
    fraction  = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, fraction))
    alloc     = min(portfolio_value * fraction, cash * 0.92)
    return alloc


def calc_shares(alloc: float, price: float) -> float:
    if alloc < 1.0 or price <= 0:
        return 0.0
    return round(alloc / price, 6)


# ── Exits ─────────────────────────────────────────────────────────────────────
def check_exits(api, positions: list, peak_prices: dict) -> tuple:
    """Check stop loss, take profit, and trailing stops. Returns (exits, updated peak_prices)."""
    exits = []

    for pos in positions:
        symbol = pos.symbol
        qty    = float(pos.qty)
        plpc   = float(pos.unrealized_plpc)
        price  = float(pos.current_price)

        # Update peak price for trailing stop
        prev_peak = peak_prices.get(symbol, float(pos.avg_entry_price))
        peak      = max(prev_peak, price)
        peak_prices[symbol] = peak

        # Hard stop loss
        if plpc <= -STOP_LOSS_PCT:
            log.warning(f"🛑 STOP LOSS {symbol}: {plpc:.1%}")
            order = place_order(api, symbol, qty, "sell", "stop_loss")
            if order:
                exits.append(order)
                peak_prices.pop(symbol, None)
            continue

        # Take profit
        if plpc >= TAKE_PROFIT_PCT:
            log.info(f"🎯 TAKE PROFIT {symbol}: {plpc:.1%}")
            order = place_order(api, symbol, qty, "sell", "take_profit")
            if order:
                exits.append(order)
                peak_prices.pop(symbol, None)
            continue

        # Trailing stop (only activates after 2% gain)
        entry = float(pos.avg_entry_price)
        if peak > entry * 1.02:
            trail_trigger = peak * (1 - TRAILING_STOP_PCT)
            if price <= trail_trigger:
                log.info(f"📉 TRAILING STOP {symbol}: peak={peak:.2f} now={price:.2f} ({plpc:.1%})")
                order = place_order(api, symbol, qty, "sell", "trailing_stop")
                if order:
                    exits.append(order)
                    peak_prices.pop(symbol, None)

    return exits, peak_prices


# ── Universe ──────────────────────────────────────────────────────────────────
def get_universe(api, state: dict, held_symbols: list) -> List[str]:
    """
    Full scan at market open (first 45 mins).
    Quick scan (held + volatile) rest of day.
    """
    now        = datetime.now(timezone.utc)
    today      = now.strftime("%Y-%m-%d")
    open_time  = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    mins_since = (now - open_time).total_seconds() / 60

    # Full scan at open OR no cache yet today
    if state.get("universe_date") != today or mins_since <= FULL_SCAN_WINDOW:
        log.info("🌍 Full universe scan...")
        try:
            universe = build_universe()
        except Exception:
            universe = FALLBACK_UNIVERSE
        state["universe_cache"] = universe
        state["universe_date"]  = today
        return universe

    # Quick scan: held positions + top volatile from cache
    log.info("⚡ Quick scan (held + volatile)...")
    cached   = state.get("universe_cache", FALLBACK_UNIVERSE)
    quick    = list(set(held_symbols + cached[:50]))
    return quick


# ── Main ──────────────────────────────────────────────────────────────────────
def run_bot():
    log.info("=" * 60)
    log.info(f"🤖 AlgoTrader v2 | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    log.info("=" * 60)

    api   = get_api()
    state = load_state()

    account = get_account(api)
    log.info(f"💰 Portfolio: ${account['portfolio_value']:,.2f} | Cash: ${account['cash']:,.2f}")

    if account["status"] != "ACTIVE":
        log.error("Account not active — aborting")
        return

    # Reset daily counter
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("trade_date") != today:
        state["daily_trades"] = 0
        state["trade_date"]   = today

    # Market regime filter
    bullish_market = get_spy_trend(api)

    # Positions
    positions   = api.list_positions()
    held        = {p.symbol: p for p in positions}
    n_positions = len(positions)

    # Trailing stop & exits
    peak_prices = state.get("peak_prices", {})
    exits, peak_prices = check_exits(api, positions, peak_prices)
    state["peak_prices"] = peak_prices

    if exits:
        state["trades"].extend(exits)
        state["daily_trades"] += len(exits)
        positions   = api.list_positions()
        held        = {p.symbol: p for p in positions}
        n_positions = len(positions)

    # Universe
    universe = get_universe(api, state, list(held.keys()))
    log.info(f"🔍 Scanning {len(universe)} symbols...")

    # Fetch bars from Alpaca
    all_signals = {}
    all_buys    = []
    all_sells   = []
    batches     = [universe[i:i+BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        log.info(f"📡 Batch {i+1}/{len(batches)}")
        bars = fetch_bars_alpaca(api, batch)

        for symbol, df in bars.items():
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
        qty   = float(held[symbol].qty)
        order = place_order(api, symbol, qty, "sell", f"signal_sell(score={sig['sell_score']})")
        if order:
            state["trades"].append(order)
            state["daily_trades"] += 1
            held.pop(symbol)
            n_positions -= 1
            peak_prices.pop(symbol, None)

    # Execute buys (only if market is bullish)
    slots    = MAX_POSITIONS - n_positions
    cash     = account["cash"]
    min_cash = account["portfolio_value"] * MIN_CASH_BUFFER

    if not bullish_market:
        log.info("📉 Bear market regime — skipping new buys")
    elif slots > 0 and all_buys:
        # Sort by Kelly-adjusted score
        all_buys.sort(
            key=lambda x: x["buy_score"] * (1 + x.get("kelly", 0.5)),
            reverse=True
        )
        for sig in all_buys[:slots]:
            symbol = sig["symbol"]
            price  = sig["price"]
            kelly  = sig.get("kelly", 0.5)

            if price < 0.50:
                continue

            alloc = kelly_position_size(account["portfolio_value"], cash, kelly)
            qty   = calc_shares(alloc, price)

            if qty == 0.0:
                log.info(f"⚠️  Skip {symbol} — insufficient funds")
                continue

            cost = qty * price
            if (cash - cost) < min_cash:
                log.info(f"⚠️  Skip {symbol} — low cash buffer")
                continue

            order = place_order(
                api, symbol, qty, "buy",
                f"signal_buy(score={sig['buy_score']},kelly={kelly:.2f})"
            )
            if order:
                order["price"] = price
                state["trades"].append(order)
                state["daily_trades"] += 1
                peak_prices[symbol] = price  # initialise trailing stop peak
                cash -= cost
                n_positions += 1
    else:
        if slots <= 0:
            log.info(f"No buys — positions full ({n_positions}/{MAX_POSITIONS})")
        else:
            log.info(f"No buys — no candidates found")

    state["peak_prices"] = peak_prices
    state["signals"]     = dict(list(all_signals.items())[:100])
    state["last_run"]    = datetime.now().isoformat()
    state["account"]     = account
    save_state(state)

    log.info(f"✅ Done | Scanned={len(all_signals)} | Daily trades={state['daily_trades']}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_bot()
