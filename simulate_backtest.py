"""
Simulated Backtest with Synthetic BTC Data
===========================================
Generates realistic BTC/USDT 1m candles using Geometric Brownian Motion
calibrated to BTC's historical volatility (~60% annualised), then runs the
scalping strategy on them.

Usage:
    python simulate_backtest.py --days 7 --capital 500 --seed 42
"""

import argparse
import logging
import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from crypto_scalping_bot import Config, TechnicalAnalysis, RiskManager, Position

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic OHLCV generator
# ---------------------------------------------------------------------------
def generate_ohlcv(
    days: int,
    start_price: float = 95_000.0,
    annual_vol: float = 0.65,   # BTC ~65% annualised volatility
    annual_drift: float = 0.30, # mild upward drift
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    minutes = days * 24 * 60
    dt = 1 / (365 * 24 * 60)  # one minute in years

    mu = annual_drift
    sigma = annual_vol

    # Simulate log-returns
    z = np.random.standard_normal(minutes)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    close_prices = start_price * np.exp(np.cumsum(log_returns))
    close_prices = np.insert(close_prices, 0, start_price)[:-1]

    # Build OHLCV candles from close series
    records = []
    base_ts = datetime.now(timezone.utc) - timedelta(days=days)
    for i, close in enumerate(close_prices):
        ts = base_ts + timedelta(minutes=i)
        # intra-bar range driven by volatility
        intra_vol = close * sigma * math.sqrt(dt) * np.random.uniform(0.5, 2.5)
        high = close + abs(intra_vol)
        low = close - abs(intra_vol)
        open_ = close_prices[i - 1] if i > 0 else close
        volume = np.random.uniform(0.5, 5.0) * (1 + 0.5 * abs(z[i]))
        records.append((ts, open_, high, low, close, volume))

    df = pd.DataFrame(records, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.set_index("timestamp")
    return df


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, initial_capital: float) -> dict:
    df = TechnicalAnalysis.compute(df)
    risk = RiskManager(initial_capital)
    position = None
    trade_log = []
    current_day = None

    for i in range(Config.CANDLES_REQUIRED, len(df)):
        row = df.iloc[i]
        atr = float(row["atr"])
        candle_ts = df.index[i]

        # Simulate daily reset using candle timestamp
        candle_day = candle_ts.day
        if candle_day != current_day:
            current_day = candle_day
            risk._day = candle_day
            risk.daily_start_balance = risk.balance
            risk.daily_trades = 0
            risk.daily_pnl = 0.0

        if position is not None:
            reason = position.should_close(float(row["high"]), float(row["low"]))
            if reason:
                exit_price = position.tp if reason == "take_profit" else position.sl
                pnl = position.pnl(exit_price)
                risk.record_trade(pnl)
                trade_log.append(
                    {
                        "open_time": position.opened_at,
                        "close_time": candle_ts,
                        "side": position.side,
                        "entry": position.entry,
                        "exit": exit_price,
                        "qty": position.qty,
                        "pnl": round(pnl, 4),
                        "reason": reason,
                        "balance_after": round(risk.balance, 2),
                    }
                )
                position = None
            continue

        if not risk.can_trade():
            continue

        signal_val = int(df.iloc[i - 1]["signal"])
        if signal_val == 0:
            continue

        close = float(row["close"])
        side = "buy" if signal_val == 1 else "sell"
        qty = risk.position_size(close, atr)
        if qty <= 0:
            continue

        sl = risk.stop_loss(close, side, atr)
        tp = risk.take_profit(close, sl, side)
        position = Position(side, close, qty, sl, tp)
        position.opened_at = candle_ts

    trades_df = pd.DataFrame(trade_log)

    equity_curve = [initial_capital]
    if not trades_df.empty:
        equity_curve = [initial_capital] + trades_df["balance_after"].tolist()

    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    return {
        "initial_capital": initial_capital,
        "final_balance": risk.balance,
        "total_pnl": risk.total_pnl,
        "return_pct": risk.total_pnl / initial_capital * 100,
        "total_trades": risk.wins + risk.losses,
        "win_rate": risk.win_rate,
        "wins": risk.wins,
        "losses": risk.losses,
        "max_drawdown": max_dd,
        "trades": trades_df,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(result: dict, days: int):
    trades = result["trades"]
    proj_monthly = result["return_pct"] / days * 30
    proj_2m = result["initial_capital"] * ((1 + result["return_pct"] / 100) ** (60 / days))

    print("\n" + "=" * 60)
    print("  BACKTEST REPORT — Synthetic BTC/USDT 1m")
    print(f"  Period: {days} days   |   Capital: ${result['initial_capital']:,.2f}")
    print("=" * 60)
    print(f"  Final balance     : ${result['final_balance']:,.2f}")
    print(f"  Total PnL         : ${result['total_pnl']:+,.2f}  ({result['return_pct']:+.2f}%)")
    print(f"  Total trades      : {result['total_trades']}")
    print(f"  Win rate          : {result['win_rate']:.1%}  ({result['wins']}W / {result['losses']}L)")

    if not trades.empty:
        print(f"  Avg PnL/trade     : ${trades['pnl'].mean():+.4f}")
        print(f"  Best trade        : ${trades['pnl'].max():+.4f}")
        print(f"  Worst trade       : ${trades['pnl'].min():+.4f}")
        print(f"  Max drawdown      : {result['max_drawdown']:.2f}%")

    print("-" * 60)
    print(f"  Projected return/month : {proj_monthly:+.1f}%")
    print(f"  Projected balance @2mo : ${proj_2m:,.2f}  (if same rate holds)")
    print("=" * 60)

    if not trades.empty:
        print("\nLast 10 trades:")
        display_cols = ["open_time", "side", "entry", "exit", "pnl", "reason", "balance_after"]
        print(trades[display_cols].tail(10).to_string(index=False))
        trades.to_csv("backtest_trades.csv", index=False)
        print("\nFull trade log saved to backtest_trades.csv")

    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated BTC Scalping Backtest")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--capital", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-price", type=float, default=95_000.0)
    args = parser.parse_args()

    log.info("Generating %d days of synthetic BTC/USDT 1m candles…", args.days)
    df = generate_ohlcv(args.days, start_price=args.start_price, seed=args.seed)
    log.info("Running backtest on %d candles…", len(df))
    result = run_backtest(df, args.capital)
    print_report(result, args.days)
