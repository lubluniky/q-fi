# Makefile for Pump-and-Dump Detection System
# Convenient commands for building, testing, and running

.PHONY: help build test clean run run-test run-small run-full install check format lint

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Pump-and-Dump Detection System - Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install          Install dependencies and build Rust module"
	@echo "  make build            Build Rust module in release mode"
	@echo "  make build-dev        Build Rust module in dev mode (faster)"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test             Run all tests"
	@echo "  make test-rust        Run Rust tests only"
	@echo "  make check            Verify installation"
	@echo ""
	@echo "$(GREEN)Running:$(NC)"
	@echo "  make run-test         Test single symbol (BTC/USDT)"
	@echo "  make run-small        Small backtest (10 symbols)"
	@echo "  make run-medium       Medium backtest (100 symbols)"
	@echo "  make run-full         Full backtest (1000 symbols)"
	@echo "  make run              Alias for run-medium"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean            Remove cache and build artifacts"
	@echo "  make clean-data       Remove cached data only"
	@echo "  make clean-results    Remove results only"
	@echo "  make clean-all        Remove everything (cache, data, results, venv)"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make format           Format Rust code"
	@echo "  make lint             Run Rust linter (clippy)"
	@echo "  make benchmark        Run performance benchmarks"
	@echo ""

install: ## Install all dependencies and build
	@echo "$(BLUE)[1/4] Creating virtual environment...$(NC)"
	python3 -m venv venv
	@echo "$(BLUE)[2/4] Installing Python dependencies...$(NC)"
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "$(BLUE)[3/4] Building Rust module...$(NC)"
	. venv/bin/activate && maturin develop --release
	@echo "$(BLUE)[4/4] Creating directories...$(NC)"
	mkdir -p data results/best_pumps
	@echo "$(GREEN)✓ Installation complete!$(NC)"

build: ## Build Rust module (release mode)
	@echo "$(BLUE)Building Rust module (release)...$(NC)"
	maturin develop --release
	@echo "$(GREEN)✓ Build complete!$(NC)"

build-dev: ## Build Rust module (dev mode - faster)
	@echo "$(BLUE)Building Rust module (dev)...$(NC)"
	maturin develop
	@echo "$(GREEN)✓ Build complete!$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running test suite...$(NC)"
	python test.py
	@echo "$(GREEN)✓ Tests passed!$(NC)"

test-rust: ## Run Rust tests
	@echo "$(BLUE)Running Rust tests...$(NC)"
	cargo test
	@echo "$(GREEN)✓ Rust tests passed!$(NC)"

check: ## Verify installation
	@echo "$(BLUE)Checking installation...$(NC)"
	@python -c "import quant_engine; print('$(GREEN)✓ quant_engine module loaded$(NC)')" || \
		(echo "$(RED)✗ quant_engine not found. Run 'make build'$(NC)" && exit 1)
	@python -c "import ccxt; print('$(GREEN)✓ ccxt installed$(NC)')" || \
		(echo "$(RED)✗ ccxt not found. Run 'make install'$(NC)" && exit 1)
	@echo "$(GREEN)✓ All checks passed!$(NC)"

run-test: check ## Test single symbol (BTC/USDT)
	@echo "$(BLUE)Testing BTC/USDT...$(NC)"
	python python/main.py --single-symbol BTC/USDT

run-small: check ## Small backtest (5 spot + 5 futures)
	@echo "$(BLUE)Running small backtest (10 symbols)...$(NC)"
	python python/main.py --spot-limit 5 --futures-limit 5

run-medium: check ## Medium backtest (50 spot + 50 futures)
	@echo "$(BLUE)Running medium backtest (100 symbols)...$(NC)"
	python python/main.py --spot-limit 50 --futures-limit 50

run-full: check ## Full backtest (500 spot + 500 futures)
	@echo "$(BLUE)Running full backtest (1000 symbols)...$(NC)"
	@echo "$(YELLOW)This may take 10-15 minutes...$(NC)"
	python python/main.py

run: run-medium ## Alias for run-medium

clean-data: ## Remove cached data
	@echo "$(YELLOW)Removing cached data...$(NC)"
	rm -rf data/*.parquet
	@echo "$(GREEN)✓ Cache cleared$(NC)"

clean-results: ## Remove results
	@echo "$(YELLOW)Removing results...$(NC)"
	rm -rf results/*
	mkdir -p results/best_pumps
	@echo "$(GREEN)✓ Results cleared$(NC)"

clean: ## Remove cache and build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf target/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf python/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.so" -delete
	@echo "$(GREEN)✓ Clean complete$(NC)"

clean-all: clean clean-data clean-results ## Remove everything
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	rm -rf venv/
	@echo "$(GREEN)✓ Everything cleaned$(NC)"

format: ## Format Rust code
	@echo "$(BLUE)Formatting Rust code...$(NC)"
	cargo fmt
	@echo "$(GREEN)✓ Formatting complete$(NC)"

lint: ## Run Rust linter
	@echo "$(BLUE)Running clippy...$(NC)"
	cargo clippy -- -D warnings
	@echo "$(GREEN)✓ Lint complete$(NC)"

benchmark: check ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmark...$(NC)"
	@echo "$(YELLOW)Testing detection speed on 10,000 candles...$(NC)"
	@python -c "\
	import quant_engine; \
	import time; \
	import numpy as np; \
	n = 10000; \
	opens = [100 + np.random.normal(0, 5) for _ in range(n)]; \
	highs = [o + np.random.uniform(1, 3) for o in opens]; \
	volumes = [10000 + np.random.normal(0, 1000) for _ in range(n)]; \
	start = time.time(); \
	result = quant_engine.detect_anomalies(opens, highs, volumes); \
	elapsed = time.time() - start; \
	print(f'$(GREEN)✓ Processed {n} candles in {elapsed*1000:.2f}ms$(NC)'); \
	print(f'  Speed: {n/elapsed:.0f} candles/sec'); \
	print(f'  Detected: {len(result)} anomalies')"

# Shortcuts
b: build ## Shortcut for build
t: test ## Shortcut for test
r: run ## Shortcut for run
c: clean ## Shortcut for clean

# Development helpers
dev-setup: install test ## Complete dev setup
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo ""
	@echo "Try:"
	@echo "  make run-test    # Test single symbol"
	@echo "  make run-small   # Small backtest"

# Watch mode (requires entr)
watch: ## Watch Rust files and rebuild on changes
	@echo "$(BLUE)Watching for changes...$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(NC)"
	find src -name "*.rs" | entr -c make build-dev

# Generate documentation
docs: ## Generate Rust documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	cargo doc --no-deps --open
	@echo "$(GREEN)✓ Documentation generated$(NC)"

# Quick status check
status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo ""
	@echo "$(GREEN)Rust:$(NC)"
	@rustc --version 2>/dev/null || echo "  $(RED)Not installed$(NC)"
	@cargo --version 2>/dev/null || echo "  $(RED)Not installed$(NC)"
	@echo ""
	@echo "$(GREEN)Python:$(NC)"
	@python3 --version 2>/dev/null || echo "  $(RED)Not installed$(NC)"
	@echo ""
	@echo "$(GREEN)Module Status:$(NC)"
	@python -c "import quant_engine; print('  ✓ quant_engine loaded')" 2>/dev/null || echo "  $(RED)✗ quant_engine not built$(NC)"
	@echo ""
	@echo "$(GREEN)Data:$(NC)"
	@echo "  Cached files: $$(find data -name '*.parquet' 2>/dev/null | wc -l)"
	@echo ""
	@echo "$(GREEN)Results:$(NC)"
	@echo "  Plots: $$(find results/best_pumps -name '*.png' 2>/dev/null | wc -l)"
	@echo ""

# Release build with native optimizations
release: ## Build with maximum optimizations
	@echo "$(BLUE)Building with native CPU optimizations...$(NC)"
	RUSTFLAGS="-C target-cpu=native" maturin develop --release
	@echo "$(GREEN)✓ Optimized build complete!$(NC)"

.SILENT: help
