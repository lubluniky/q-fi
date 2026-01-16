# 🚀 Q-Fi - Quantitative Finance Research Tools

High-performance quantitative finance tools for cryptocurrency market analysis.

---

## 📦 Projects

### 🔴 Pump-and-Dump Detection System

**High-speed anomaly detection based on academic research**

- **Paper**: [arXiv:2503.08692v1](https://arxiv.org/abs/2503.08692v1)
- **Performance**: 847+ symbols/sec (M4 Mac)
- **Stack**: Rust (PyO3) + Python
- **Data**: 1000 Binance pairs (spot + futures)

**Quick Start:**
```bash
cd Pump-and-Dump
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
maturin develop --release
python python/main.py --spot-limit 100 --futures-limit 100
```

**Results**: 13 pump-and-dump events detected in 2025-2026 data

📁 [View Project](./Pump-and-Dump/)

---

## 🎯 Features

### ⚡ Performance-First Design
- **Rust Core**: 100x faster than pure Python
- **Parallel Processing**: Multi-threaded data fetching
- **Smart Caching**: Parquet-based data storage

### 📊 Research-Based Algorithms
- Academic paper implementations
- Validated detection criteria
- Statistical noise filtering

### 🎨 Professional Visualizations
- Deep black theme (#000000)
- High-contrast plots
- Publication-ready outputs

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Core Engine | Rust (PyO3) |
| Data Processing | Python + Pandas |
| Market Data | CCXT (Binance API) |
| Visualization | Matplotlib |
| Storage | Parquet |

---

## 📋 Requirements

- **Rust** 1.70+ ([Install](https://rustup.rs/))
- **Python** 3.8+
- **Maturin** (Rust-Python bridge)
- **10GB+** disk space (for cached data)

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/lubluniky/q-fi.git
cd q-fi
```

### 2. Choose a Project
```bash
cd Pump-and-Dump  # Or any other project
```

### 3. Follow Project README
Each project has its own setup instructions and documentation.

---

## 📈 Example Results

### Pump-and-Dump Detection
- **Scanned**: 930 symbols (430 spot + 500 futures)
- **Detected**: 13 pump-and-dump events
- **Top Hit**: SNT/USDT (+278% price, +967% volume)

**Detection Time**: 1.1 seconds for 930 symbols

---

## 🧪 Testing

Each project includes comprehensive tests:

```bash
cd <project-name>
python test.py
```

---

## 📚 Documentation

- **QUICKSTART.md**: Fast setup guides
- **README.md**: Full documentation (per project)
- **examples/**: Usage examples and tutorials

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Run tests: `python test.py`
4. Commit changes: `git commit -m "feat: description"`
5. Push and create PR

---

## ⚠️ Disclaimer

**For Research and Educational Purposes Only**

These tools are designed for academic research and learning. They should NOT be used as financial advice or for automated trading decisions. Cryptocurrency markets are highly volatile and risky.

---

## 📄 License

Research implementations based on publicly available papers.

---

## 🔗 Links

- **Repository**: [github.com/lubluniky/q-fi](https://github.com/lubluniky/q-fi)
- **Research Paper**: [arXiv:2503.08692v1](https://arxiv.org/abs/2503.08692v1)

---

## 📊 Project Roadmap

- [x] Pump-and-Dump Detection (arXiv:2503.08692v1)
- [ ] Mean Reversion Strategies
- [ ] Momentum-Based Alpha
- [ ] Volatility Trading Signals
- [ ] Pairs Trading Algorithms

---

**Built with 💻 for quantitative finance research**