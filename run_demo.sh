#!/usr/bin/env bash
# ==============================================================
# SentinelAI — One-Click Presentation & Demo Launcher
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================="
echo "  🛡️  SentinelAI: Prompt Injection Firewall Demo Launcher"
echo "=============================================================="
echo ""

# Activate virtual environment if present
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Launch Flask application
cd "$SCRIPT_DIR/app"
python app.py
