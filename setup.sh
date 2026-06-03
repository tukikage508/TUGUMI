#!/bin/bash
# TUGUMI Self-Autonomous AI Agent Setup Script

set -e

echo "=================================================="
echo "TUGUMI - Self-Autonomous AI Agent Setup"
echo "=================================================="

# Create necessary directories
echo "[1/5] Creating directories..."
mkdir -p /storage/emulated/0/TUGUMIDesk
mkdir -p logs
mkdir -p data
mkdir -p cache

# Install Python dependencies
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create initial configuration
echo "[3/5] Creating configuration files..."
cat > .env << 'EOF'
# TUGUMI Configuration
LLAMA_SERVER_URL=http://0.0.0.0:8080
LLAMA_TIMEOUT=300
LOG_LEVEL=INFO
OUTPUT_DIR=/storage/emulated/0/TUGUMIDesk
LOG_DIR=./logs
CACHE_DIR=./cache
MAX_MEMORY_ITEMS=100
MAX_RETRIES=3
RETRY_DELAY=2
EOF

# Initialize logging
echo "[4/5] Initializing logging system..."
mkdir -p logs
touch logs/agent.log

# Verify llama.cpp server connectivity (optional)
echo "[5/5] Setup Complete!"
echo "=================================================="
echo "Configuration Summary:"
echo "- Local LLM Server: http://0.0.0.0:8080"
echo "- Output Directory: /storage/emulated/0/TUGUMIDesk"
echo "- Log Directory: ./logs"
echo "=================================================="
echo ""
echo "To start the agent, run:"
echo "  python main.py"
echo ""
