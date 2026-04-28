#!/usr/bin/env bash
# run_backtest.sh — install dependencies and run backtester

set -euo pipefail

echo "=== Crypto Scalping Backtester ==="

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Default: BTC/USDT, 1m candles, last 7 days, $1000 capital
# Override with args: ./run_backtest.sh --symbol ETH/USDT --days 14 --capital 5000
python backtester.py "$@"
