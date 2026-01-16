# 🚀 QUICKSTART - Pump-and-Dump Detection

Get started with pump-and-dump detection in **5 minutes**.

---

## ⚡ Fast Setup (macOS/Linux)

```bash
# 1. Navigate to project
cd Pump-and-Dump

# 2. Run automated build script
chmod +x build.sh
./build.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Run quick test
python test.py
```

---

## 🪟 Windows Setup

```powershell
# 1. Navigate to project
cd Pump-and-Dump

# 2. Install Python dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Build Rust module
maturin develop --release

# 4. Run test
python test.py
```

---

## 🎯 First Run - Test Single Symbol

```bash
# Test Bitcoin (fastest way to see results)
python python/main.py --single-symbol BTC/USDT

# Test Ethereum
python python/main.py --single-symbol ETH/USDT --market-type spot
```

**Output:**
- Console logs showing detection progress
- Plots saved to `results/best_pumps/`
- Detection report in `results/pump_dump_report.txt`

---

## 🧪 Small Backtest (10 symbols - ~2 minutes)

```bash
python python/main.py --spot-limit 5 --futures-limit 5
```

This will:
1. ✓ Fetch top 5 spot + 5 futures pairs
2. ✓ Download 1h OHLCV data (2025-2026)
3. ✓ Run Rust detection algorithm
4. ✓ Generate visualizations
5. ✓ Create summary report

---

## 🔥 Full Backtest (1000 symbols - ~10 minutes)

```bash
# Default: 500 spot + 500 futures
python python/main.py

# Custom limits
python python/main.py --spot-limit 200 --futures-limit 300
```

**Performance Tips:**
- First run downloads data (slower)
- Subsequent runs use cache (faster)
- Use `--max-workers 20` for faster downloads
- Use `--force-refresh` to update cached data

---

## 📊 View Results

After running, check:

```bash
# View summary report
cat results/pump_dump_report.txt

# View execution logs
cat results/backtest.log

# View visualizations
open results/best_pumps/
```

---

## 🛠️ Common Commands

### Test Rust Module
```bash
python -c "import quant_engine; print('✓ Module loaded')"
```

### Rebuild After Code Changes
```bash
maturin develop --release
```

### Clear Cache and Start Fresh
```bash
rm -rf data/*.parquet
python python/main.py --force-refresh --spot-limit 10
```

### Check Performance
```bash
# Time a full backtest
time python python/main.py --spot-limit 100 --futures-limit 100
```

---

## 🎨 Visualization Examples

All plots use a **deep black theme** with high contrast:

- **Background:** Pure black (#000000)
- **Price Line:** Off-white (#E0E0E0)
- **Anomaly Marker:** Bright red (#FF3333)
- **Grid:** Very subtle (#1A1A1A)

Each plot shows:
- Top panel: Price + 12h MA + pump marker
- Bottom panel: Volume bars + MA overlay
- Title: Symbol, timestamp, price/volume spike %

---

## 🔧 Troubleshooting

### "quant_engine module not found"
```bash
# Solution: Build the Rust module
maturin develop --release
```

### "Rate limit exceeded"
```bash
# Solution: Reduce parallel workers
python python/main.py --max-workers 5
```

### "Out of memory"
```bash
# Solution: Process fewer symbols
python python/main.py --spot-limit 50 --futures-limit 50
```

### Cached data is outdated
```bash
# Solution: Force refresh
python python/main.py --force-refresh
```

---

## 📈 Understanding Results

### Report Statistics

```
Total Symbols Scanned: 1000
Symbols with Detected Pumps: 234    ← 23.4% hit rate
Total Anomalies Detected: 456       ← Multiple pumps per symbol
Average Anomalies per Symbol: 0.46
```

### Anomaly Format

```
BTC/USDT (SPOT) - 3 pump(s)
  - 2025-03-15 14:00 | Price: +127.3% | Volume: +543.2%
     ↑              ↑         ↑                ↑
  Symbol      Timestamp   Price spike    Volume spike
```

### Detection Criteria (ALL must be met)

1. **Price Spike:** High > 1.9× MA(Open, 12h) → **90%+ increase**
2. **Volume Spike:** Volume > 5× MA(Volume, 12h) → **400%+ increase**
3. **Noise Filter:** Volume between statistical bounds → **Excludes false positives**

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| **Build** | `maturin develop --release` |
| **Test** | `python test.py` |
| **Single Symbol** | `python python/main.py --single-symbol BTC/USDT` |
| **Small Test** | `python python/main.py --spot-limit 5 --futures-limit 5` |
| **Full Backtest** | `python python/main.py` |
| **Force Refresh** | `python python/main.py --force-refresh` |
| **View Report** | `cat results/pump_dump_report.txt` |

---

## 🚦 Next Steps

### 1. Customize Parameters
Edit `config.py` to change:
- Detection thresholds
- Lookback windows
- Visualization colors
- Output formats

### 2. Modify Algorithm
Edit `src/lib.rs` to:
- Adjust detection logic
- Add new indicators
- Change window sizes

**After changes:**
```bash
maturin develop --release
```

### 3. Integrate with Trading
Use detected pumps for:
- Backtesting trading strategies
- Risk management
- Market analysis
- Research

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | Core Rust detection engine |
| `python/main.py` | Orchestration & backtesting |
| `python/data_ingestion.py` | Binance data fetching |
| `python/visualization.py` | Plot generation |
| `config.py` | Configuration settings |
| `test.py` | Validation tests |

---

## ⚠️ Important Notes

1. **First run is slow** (downloads data) - subsequent runs are fast (uses cache)
2. **Respect Binance rate limits** - use `--max-workers 10` or less
3. **Results are for research only** - not financial advice
4. **Check logs** if something fails - they're detailed

---

## 💡 Pro Tips

### Speed Up Development
```bash
# Use cached data during development
python python/main.py --spot-limit 10

# Test on fast symbols (BTC, ETH)
python python/main.py --single-symbol BTC/USDT
```

### Maximize Performance
```bash
# Build with full optimizations
RUSTFLAGS="-C target-cpu=native" maturin develop --release

# Use more workers (if you have good bandwidth)
python python/main.py --max-workers 20
```

### Debug Mode
```bash
# Enable detailed logging
python python/main.py --spot-limit 5 2>&1 | tee debug.log
```

---

## ✅ Success Checklist

- [ ] Rust installed (`rustc --version`)
- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Rust module built (`maturin develop --release`)
- [ ] Tests passing (`python test.py`)
- [ ] Single symbol works (`python python/main.py --single-symbol BTC/USDT`)
- [ ] Results generated in `results/` directory

---

## 🆘 Need Help?

1. **Read the full README:** `README.md`
2. **Check test output:** `python test.py`
3. **View logs:** `cat results/backtest.log`
4. **Verify Rust module:** `python -c "import quant_engine; print('OK')"`

---

## 🎉 You're Ready!

```bash
# Run your first backtest
python python/main.py --spot-limit 50 --futures-limit 50

# Watch the magic happen! 🚀
```

**Expected Output:**
```
================================================================================
PUMP-AND-DUMP DETECTION BACKTEST
Algorithm: arXiv:2503.08692v1 'Best Setting'
================================================================================

[STEP 1] Fetching top liquid symbols from Binance...
[STEP 2] Downloading OHLCV data (1h candles, 2025-2026)...
[STEP 3] Running Rust-powered anomaly detection...
[STEP 4] Generating summary report...
[STEP 5] Visualizing top 20 pump-and-dump events...

✅ BACKTEST COMPLETE
```

**Happy pump hunting! 🔍📈**