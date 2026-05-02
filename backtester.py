"""
Backtester for the Scalping Bot
================================
Downloads historical OHLCV data from Binance and simulates the strategy
without spending real money.

Usage:
    python backtester.py --symbol BTC/USDT --timeframe 1m --days 30

Dependencies:
    pip install ccxt pandas ta python-dotenv
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone

import ssl
import urllib3
import requests
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Force all requests sessions to skip SSL verification (self-signed proxy certs)
_orig_session_init = requests.Session.__init__
def _patched_session_init(self, *args, **kwargs):
    _orig_session_init(self, *args, **kwargs)
    self.verify = False
requests.Session.__init__ = _patched_session_init

import ccxt
import pandas as pd

from crypto_scalping_bot import Config, TechnicalAnalysis, RiskManager, Position

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def fetch_history(symbol: str, timeframe: str, since_days: int) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.session.verify = False  # allow self-signed certs in sandboxed environments
    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp() * 1000
    )
    all_ohlcv = []
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not chunk:
            break
        all_ohlcv.extend(chunk)
        since_ms = chunk[-1][0] + 1
        if len(chunk) < 1000:
            break

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").drop_duplicates()
    log.info("Fetched %d candles for %s [%s].", len(df), symbol, timeframe)
    return df


def run_backtest(df: pd.DataFrame, initial_capital: float) -> dict:
    df = TechnicalAnalysis.compute(df)
    risk = RiskManager(initial_capital)
    position = None
    trade_log = []

    for i in range(Config.CANDLES_REQUIRED, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        atr = float(row["atr"])

        if position is not None:
            reason = position.should_close(float(row["high"]), float(row["low"]))
            if reason:
                exit_price = position.tp if reason == "take_profit" else position.sl
                pnl = position.pnl(exit_price)
                risk.record_trade(pnl)
                trade_log.append(
                    {
                        "open_time": position.opened_at,
                        "close_time": df.index[i],
                        "side": position.side,
                        "entry": position.entry,
                        "exit": exit_price,
                        "qty": position.qty,
                        "pnl": pnl,
                        "reason": reason,
                    }
                )
                position = None
            continue

        if not risk.can_trade():
            continue

        signal_val = int(df.iloc[i - 1]["signal"])
        if signal_val == 0:
            continue

        side = "buy" if signal_val == 1 else "sell"
        qty = risk.position_size(price, atr)
        if qty <= 0:
            continue

        sl = risk.stop_loss(price, side, atr)
        tp = risk.take_profit(price, sl, side)
        position = Position(side, price, qty, sl, tp)
        position.opened_at = df.index[i]

    trades_df = pd.DataFrame(trade_log)
    result = {
        "initial_capital": initial_capital,
        "final_balance": risk.balance,
        "total_pnl": risk.total_pnl,
        "return_pct": risk.total_pnl / initial_capital * 100,
        "total_trades": risk.wins + risk.losses,
        "win_rate": risk.win_rate,
        "wins": risk.wins,
        "losses": risk.losses,
        "trades": trades_df,
    }
    return result


def print_report(result: dict):
    trades = result["trades"]
    print("\n" + "=" * 55)
    print("  BACKTEST REPORT")
    print("=" * 55)
    print(f"  Initial capital : ${result['initial_capital']:,.2f}")
    print(f"  Final balance   : ${result['final_balance']:,.2f}")
    print(f"  Total PnL       : ${result['total_pnl']:+,.2f}  ({result['return_pct']:+.2f}%)")
    print(f"  Total trades    : {result['total_trades']}")
    print(f"  Win rate        : {result['win_rate']:.1%}  ({result['wins']}W / {result['losses']}L)")

    if not trades.empty:
        print(f"  Avg PnL/trade   : ${trades['pnl'].mean():+.2f}")
        print(f"  Best trade      : ${trades['pnl'].max():+.2f}")
        print(f"  Worst trade     : ${trades['pnl'].min():+.2f}")
        # Max drawdown
        equity = result["initial_capital"] + trades["pnl"].cumsum()
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        print(f"  Max drawdown    : {drawdown.min():.2f}%")
    print("=" * 55 + "\n")

    if not trades.empty:
        print("Last 10 trades:")
        print(
            trades[["close_time", "side", "entry", "exit", "pnl", "reason"]]
            .tail(10)
            .to_string(index=False)
        )
        trades.to_csv("backtest_trades.csv", index=False)
        print("\nFull trade log saved to backtest_trades.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Scalping Backtester")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--capital", type=float, default=1000.0)
    args = parser.parse_args()

    df = fetch_history(args.symbol, args.timeframe, args.days)
    result = run_backtest(df, args.capital)
    print_report(result)
