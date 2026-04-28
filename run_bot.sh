#!/usr/bin/env bash
# run_bot.sh — install dependencies and start the scalping bot

set -euo pipefail

echo "=== Crypto Scalping Bot ==="

# 1. Create virtualenv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv .venv
fi

# 2. Activate and install dependencies
echo "[2/3] Installing dependencies..."
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 3. Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "  ⚠  No .env file found."
    echo "  Copy .env.example to .env and set your values:"
    echo "      cp .env.example .env"
    echo ""
    exit 1
fi

# 4. Launch bot
echo "[3/3] Starting bot... (Ctrl+C to stop)"
echo ""
python crypto_scalping_bot.py
