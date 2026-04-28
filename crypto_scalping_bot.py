"""
Crypto Scalping Bot
===================
Strategy: Multi-indicator confluence scalping on 1m/3m candles.
Exchange: Binance (via ccxt) — swap API keys in .env to use another.
Risk:      1% of balance per trade, 2:1 reward-to-risk ratio.

⚠️  DISCLAIMER: Trading cryptocurrencies carries significant financial risk.
    Past performance does not guarantee future results. This bot is provided
    for educational purposes. Never trade with money you cannot afford to lose.

Dependencies:
    pip install ccxt pandas ta python-dotenv

Environment variables (create a .env file):
    BINANCE_API_KEY=your_key
    BINANCE_API_SECRET=your_secret
    TRADING_SYMBOL=BTC/USDT          # pair to trade
    INITIAL_CAPITAL=1000             # starting USDT
    RISK_PER_TRADE=0.01              # 1% risk per trade
    MAX_DAILY_TRADES=30
    MAX_DAILY_LOSS_PCT=0.05          # stop for the day at -5%
    DRY_RUN=true                     # set to false for live trading
"""

import os
import time
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv

try:
    import ta
except ImportError:
    sys.exit("Missing dependency: pip install ta")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scalping_bot.log"),
    ],
)
log = logging.getLogger(__name__)

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class Config:
    SYMBOL: str = os.getenv("TRADING_SYMBOL", "BTC/USDT")
    TIMEFRAME: str = os.getenv("TIMEFRAME", "1m")
    INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", 1000))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", 0.01))   # 1 %
    REWARD_RATIO: float = float(os.getenv("REWARD_RATIO", 2.0))         # TP = 2×SL
    MAX_DAILY_TRADES: int = int(os.getenv("MAX_DAILY_TRADES", 30))
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", 0.05))
    DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() != "false"

    # Technical indicator parameters
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    EMA_TREND: int = 50
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 35
    RSI_OVERBOUGHT: float = 65
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    ATR_PERIOD: int = 14
    SL_ATR_MULT: float = 1.5    # stop-loss = 1.5 × ATR

    CANDLES_REQUIRED: int = 100  # minimum candles before trading
    LOOP_SLEEP: int = 60         # seconds between iterations


# ---------------------------------------------------------------------------
# Technical Analysis
# ---------------------------------------------------------------------------
class TechnicalAnalysis:
    """Computes indicators and returns a signal: 1 (long), -1 (short), 0 (flat)."""

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        df["ema_fast"] = ta.trend.EMAIndicator(close, Config.EMA_FAST).ema_indicator()
        df["ema_slow"] = ta.trend.EMAIndicator(close, Config.EMA_SLOW).ema_indicator()
        df["ema_trend"] = ta.trend.EMAIndicator(close, Config.EMA_TREND).ema_indicator()

        df["rsi"] = ta.momentum.RSIIndicator(close, Config.RSI_PERIOD).rsi()

        macd_obj = ta.trend.MACD(close)
        df["macd"] = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_hist"] = macd_obj.macd_diff()

        bb = ta.volatility.BollingerBands(close, Config.BB_PERIOD, Config.BB_STD)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()

        df["atr"] = ta.volatility.AverageTrueRange(
            high, low, close, Config.ATR_PERIOD
        ).average_true_range()

        df["signal"] = TechnicalAnalysis._signals(df)
        return df

    @staticmethod
    def _signals(df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        long_cond = (
            (df["ema_fast"] > df["ema_slow"])           # short-term bullish cross
            & (df["close"] > df["ema_trend"])           # price above trend EMA
            & (df["rsi"] < Config.RSI_OVERBOUGHT)       # not overbought
            & (df["rsi"] > Config.RSI_OVERSOLD)         # momentum present
            & (df["macd_hist"] > 0)                     # MACD positive
            & (df["close"] > df["bb_mid"])              # above BB midline
        )

        short_cond = (
            (df["ema_fast"] < df["ema_slow"])
            & (df["close"] < df["ema_trend"])
            & (df["rsi"] > Config.RSI_OVERSOLD)
            & (df["rsi"] < Config.RSI_OVERBOUGHT)
            & (df["macd_hist"] < 0)
            & (df["close"] < df["bb_mid"])
        )

        signals[long_cond] = 1
        signals[short_cond] = -1
        return signals


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------
class RiskManager:
    def __init__(self, initial_capital: float):
        self.balance: float = initial_capital
        self.daily_start_balance: float = initial_capital
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self._day: int = datetime.now(timezone.utc).day

    def _reset_daily_if_needed(self):
        today = datetime.now(timezone.utc).day
        if today != self._day:
            self._day = today
            self.daily_start_balance = self.balance
            self.daily_trades = 0
            self.daily_pnl = 0.0
            log.info("Daily counters reset.")

    def can_trade(self) -> bool:
        self._reset_daily_if_needed()
        if self.daily_trades >= Config.MAX_DAILY_TRADES:
            log.warning("Daily trade limit reached (%d).", Config.MAX_DAILY_TRADES)
            return False
        daily_loss = self.daily_pnl / self.daily_start_balance
        if daily_loss <= -Config.MAX_DAILY_LOSS_PCT:
            log.warning("Daily loss limit hit (%.2f%%).", daily_loss * 100)
            return False
        return True

    def position_size(self, price: float, atr: float) -> float:
        """Risk a fixed % of balance; size based on ATR stop-loss distance."""
        sl_distance = atr * Config.SL_ATR_MULT
        if sl_distance <= 0:
            return 0.0
        risk_amount = self.balance * Config.RISK_PER_TRADE
        qty = risk_amount / sl_distance
        return round(qty, 6)

    def stop_loss(self, entry: float, side: str, atr: float) -> float:
        dist = atr * Config.SL_ATR_MULT
        return entry - dist if side == "buy" else entry + dist

    def take_profit(self, entry: float, sl: float, side: str) -> float:
        dist = abs(entry - sl) * Config.REWARD_RATIO
        return entry + dist if side == "buy" else entry - dist

    def record_trade(self, pnl: float):
        self.balance += pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.daily_trades += 1
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total else 0.0

    def summary(self) -> str:
        return (
            f"Balance: ${self.balance:.2f} | "
            f"Total PnL: ${self.total_pnl:+.2f} | "
            f"Win rate: {self.win_rate:.1%} "
            f"({self.wins}W/{self.losses}L) | "
            f"Daily trades: {self.daily_trades}"
        )


# ---------------------------------------------------------------------------
# Position tracker
# ---------------------------------------------------------------------------
class Position:
    def __init__(
        self,
        side: str,
        entry: float,
        qty: float,
        sl: float,
        tp: float,
    ):
        self.side = side
        self.entry = entry
        self.qty = qty
        self.sl = sl
        self.tp = tp
        self.opened_at = datetime.now(timezone.utc)

    def pnl(self, price: float) -> float:
        if self.side == "buy":
            return (price - self.entry) * self.qty
        return (self.entry - price) * self.qty

    def should_close(self, high: float, low: float) -> Optional[str]:
        if self.side == "buy":
            if low <= self.sl:
                return "stop_loss"
            if high >= self.tp:
                return "take_profit"
        else:
            if high >= self.sl:
                return "stop_loss"
            if low <= self.tp:
                return "take_profit"
        return None

    def __str__(self):
        return (
            f"{self.side.upper()} {self.qty} @ {self.entry:.4f} "
            f"SL={self.sl:.4f} TP={self.tp:.4f}"
        )


# ---------------------------------------------------------------------------
# Exchange wrapper
# ---------------------------------------------------------------------------
class ExchangeClient:
    def __init__(self):
        exchange_class = getattr(ccxt, os.getenv("EXCHANGE", "binance"))
        params: dict = {"enableRateLimit": True}
        if not Config.DRY_RUN:
            params["apiKey"] = os.getenv("BINANCE_API_KEY", "")
            params["secret"] = os.getenv("BINANCE_API_SECRET", "")
        self.exchange: ccxt.Exchange = exchange_class(params)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")

    def ticker(self, symbol: str) -> dict:
        return self.exchange.fetch_ticker(symbol)

    def place_order(self, symbol: str, side: str, qty: float, price: float) -> dict:
        if Config.DRY_RUN:
            log.info("[DRY RUN] %s %s %.6f @ %.4f", side.upper(), symbol, qty, price)
            return {"id": "dry_run", "price": price, "amount": qty}
        return self.exchange.create_order(symbol, "market", side, qty)


# ---------------------------------------------------------------------------
# Main Bot
# ---------------------------------------------------------------------------
class ScalpingBot:
    def __init__(self):
        self.config = Config()
        self.client = ExchangeClient()
        self.risk = RiskManager(Config.INITIAL_CAPITAL)
        self.position: Optional[Position] = None
        self._running = True

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *_):
        log.info("Shutting down… %s", self.risk.summary())
        self._running = False

    # ------------------------------------------------------------------
    def run(self):
        mode = "DRY RUN" if Config.DRY_RUN else "LIVE"
        log.info("=== Crypto Scalping Bot started [%s] ===", mode)
        log.info("Symbol: %s | TF: %s | Capital: $%.2f",
                 Config.SYMBOL, Config.TIMEFRAME, Config.INITIAL_CAPITAL)

        while self._running:
            try:
                self._tick()
            except ccxt.NetworkError as exc:
                log.warning("Network error: %s — retrying in 30s.", exc)
                time.sleep(30)
                continue
            except ccxt.ExchangeError as exc:
                log.error("Exchange error: %s", exc)
            except Exception as exc:  # noqa: BLE001
                log.exception("Unexpected error: %s", exc)
            time.sleep(Config.LOOP_SLEEP)

    # ------------------------------------------------------------------
    def _tick(self):
        df = self.client.fetch_ohlcv(Config.SYMBOL, Config.TIMEFRAME, limit=200)

        if len(df) < Config.CANDLES_REQUIRED:
            log.info("Waiting for enough candles (%d/%d)…", len(df), Config.CANDLES_REQUIRED)
            return

        df = TechnicalAnalysis.compute(df)
        latest = df.iloc[-2]          # use closed candle (not the live one)
        current_price = df["close"].iloc[-1]

        # -- Manage open position --
        if self.position is not None:
            reason = self.position.should_close(
                float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
            )
            if reason:
                self._close_position(current_price, reason)
            return

        # -- Look for new entry --
        if not self.risk.can_trade():
            return

        signal_val = int(latest["signal"])
        atr = float(latest["atr"])

        if signal_val == 1:
            self._open_position("buy", current_price, atr, df)
        elif signal_val == -1:
            self._open_position("sell", current_price, atr, df)
        else:
            log.debug("No signal. RSI=%.1f MACD_hist=%.6f", latest["rsi"], latest["macd_hist"])

    # ------------------------------------------------------------------
    def _open_position(self, side: str, price: float, atr: float, df: pd.DataFrame):
        qty = self.risk.position_size(price, atr)
        if qty <= 0:
            log.warning("Position size is zero — skipping trade.")
            return

        sl = self.risk.stop_loss(price, side, atr)
        tp = self.risk.take_profit(price, sl, side)

        self.client.place_order(Config.SYMBOL, side, qty, price)
        self.position = Position(side, price, qty, sl, tp)
        log.info("OPEN  %s", self.position)

    # ------------------------------------------------------------------
    def _close_position(self, price: float, reason: str):
        close_side = "sell" if self.position.side == "buy" else "buy"
        self.client.place_order(Config.SYMBOL, close_side, self.position.qty, price)

        exit_price = self.position.tp if reason == "take_profit" else self.position.sl
        pnl = self.position.pnl(exit_price)
        self.risk.record_trade(pnl)

        log.info(
            "CLOSE [%s] entry=%.4f exit=%.4f PnL=%+.2f — %s",
            reason,
            self.position.entry,
            exit_price,
            pnl,
            self.risk.summary(),
        )
        self.position = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ScalpingBot().run()
