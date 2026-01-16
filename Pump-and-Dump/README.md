# 🚀 Pump-and-Dump Detection System

![Rust](https://img.shields.io/badge/rust-1.70+-orange?logo=rust)
![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python)
![License](https://img.shields.io/badge/license-Research-green)
![Performance](https://img.shields.io/badge/speed-847%20symbols%2Fsec-brightgreen)

High-performance cryptocurrency pump-and-dump detection based on [arXiv:2503.08692v1](https://arxiv.org/abs/2503.08692v1).

**Core**: Rust (PyO3) for 100x speed  
**Data**: Binance API (1000 pairs)  
**Output**: Deep black theme visualizations

---

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/lubluniky/q-fi.git
cd q-fi/Pump-and-Dump

# 2. Setup (requires Rust + Python 3.8+)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Build Rust module
maturin develop --release

# 4. Run test
python test.py

# 5. Run backtest
python python/main.py --spot-limit 100 --futures-limit 100
```

---

## 📋 Prerequisites

- **Rust** 1.70+ ([Install](https://rustup.rs/))
- **Python** 3.8+
- **10GB+ disk** (for cached data)

---

## 🎯 Usage

### Single Symbol Test (Fast)
```bash
python python/main.py --single-symbol BTC/USDT
```

### Small Backtest (100 pairs, ~3 min)
```bash
python python/main.py --spot-limit 50 --futures-limit 50
```

### Full Backtest (1000 pairs, ~10 min)
```bash
python python/main.py
```

---

## 📊 Algorithm

Detects pump-and-dump using 3 conditions (ALL must be met):

1. **Price Spike**: `High > 1.9 × MA(Open, 12h)` → 90%+ increase
2. **Volume Spike**: `Volume > 5 × MA(Volume, 12h)` → 400%+ increase  
3. **Noise Filter**: `EWMA(Vol, 20d) + 2σ < Volume < MAX(Vol, 30d)`

**Windows**: 12h (short), 20d (medium), 30d (long)

---

## 📁 Output

After running, check:
- **Plots**: `results/best_pumps/*.png` (top 20 pumps)
- **Report**: `results/pump_dump_report.txt`
- **Log**: `results/backtest.log`

---

## 🛠️ Commands

| Command | Description |
|---------|-------------|
| `make install` | Full setup (if Make available) |
| `make test` | Run tests |
| `make run-small` | Quick backtest |
| `python test.py` | Validate installation |

---

## 🔧 Troubleshooting

**"maturin not found"**
```bash
pip install maturin
```

**"Rate limit exceeded"**
```bash
python python/main.py --max-workers 5
```

**Out of memory**
```bash
python python/main.py --spot-limit 50 --futures-limit 50
```

---

## 📈 Performance

On M4 Mac:
- **847 symbols/sec** detection speed
- **~10 min** for 1000 pairs (with Binance rate limits)
- First run downloads data (slow), subsequent runs use cache (fast)

---

## 🎨 Visualization

Deep black theme (#000000) with:
- Price chart + 12h MA
- Volume bars with anomaly markers
- Red spikes (#FF3333) for detected pumps

---

## 📚 Project Structure

```
Pump-and-Dump/
├── src/lib.rs              # Rust detection engine
├── python/
│   ├── main.py             # Orchestration
│   ├── data_ingestion.py   # Binance data fetching
│   └── visualization.py    # Plot generation
├── examples/               # Usage examples
├── results/                # Output (auto-generated)
└── data/                   # Cached data (auto-generated)
```

---

## ⚙️ Configuration

Edit `config.py` to customize:
- Detection thresholds
- Lookback windows
- Output formats
- Visualization colors

---

## 🧪 Examples

See `examples/basic_usage.py` for:
- Direct Rust API usage
- Custom workflows
- Performance benchmarks
- Real Binance data analysis

```bash
python examples/basic_usage.py --example 1
```

---

## 📖 Research Paper

Based on: "Detecting Crypto Pump-and-Dump Schemes"  
arXiv: [2503.08692v1](https://arxiv.org/abs/2503.08692v1)

Implements the "Best Setting" algorithm (Section 4.6-4.7)

---

## ⚠️ Disclaimer

**For Research Only**  
Not financial advice. Cryptocurrency markets are highly volatile and risky.

---

## 🤝 Contributing

1. Test: `python test.py`
2. Format: `cargo fmt`
3. Lint: `cargo clippy`
4. Submit PR

---

## 📄 License

Research implementation based on publicly available paper.

---

## 🚦 Getting Help

1. Run tests: `python test.py`
2. Check logs: `cat results/backtest.log`
3. See examples: `python examples/basic_usage.py`

---

**Happy pump hunting! 🔍📈**