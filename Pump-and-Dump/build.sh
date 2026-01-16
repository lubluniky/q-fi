#!/bin/bash
# Build script for Pump-and-Dump Detection System

set -e  # Exit on error

echo "======================================================================"
echo "Pump-and-Dump Detection System - Build Script"
echo "======================================================================"
echo ""

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust is not installed!"
    echo "Please install Rust from: https://rustup.rs/"
    exit 1
fi

echo "✓ Rust version: $(rustc --version)"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo ""
    echo "[1/4] Using existing virtual environment"
fi

# Activate virtual environment
echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install Python dependencies
echo ""
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Python dependencies installed"

# Build Rust module
echo ""
echo "[4/4] Building Rust module (this may take a few minutes)..."
maturin develop --release
echo "✓ Rust module built successfully"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data
mkdir -p results/best_pumps
echo "✓ Directories created"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import quant_engine; print('✓ quant_engine module loaded successfully')" || {
    echo "❌ Failed to import quant_engine module"
    exit 1
}

echo ""
echo "======================================================================"
echo "✅ Build completed successfully!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run a quick test:"
echo "     python python/main.py --single-symbol BTC/USDT"
echo ""
echo "  3. Run full backtest:"
echo "     python python/main.py --spot-limit 100 --futures-limit 100"
echo ""
echo "  4. View results in:"
echo "     - results/best_pumps/          (visualizations)"
echo "     - results/pump_dump_report.txt (summary)"
echo ""
echo "======================================================================"
