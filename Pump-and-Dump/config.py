"""
Configuration File for Pump-and-Dump Detection System
Customize parameters, thresholds, and settings here
"""

# ============================================================================
# DETECTION ALGORITHM PARAMETERS
# ============================================================================

# Lookback Windows (in hours)
W_SHORT = 12  # Short-term moving average window
W_MEDIUM = 480  # Medium-term EWMA window (20 days)
W_LONG = 720  # Long-term rolling max window (30 days)

# Detection Thresholds
PRICE_THRESHOLD = 1.90  # Price must be > 1.9x the 12h MA (90% increase)
VOLUME_THRESHOLD = 5.0  # Volume must be > 5x the 12h MA (400% increase)
VOLATILITY_SIGMA = 2.0  # Number of standard deviations for noise filter

# ============================================================================
# DATA FETCHING PARAMETERS
# ============================================================================

# Binance API Configuration
SPOT_LIMIT = 500  # Number of top spot symbols to fetch
FUTURES_LIMIT = 500  # Number of top futures symbols to fetch
MAX_WORKERS = 10  # Parallel workers for data fetching

# Date Range (for historical data)
START_DATE = "2025-01-01"  # Start date for backtesting
END_DATE = None  # End date (None = current date)

# Timeframe
TIMEFRAME = "1h"  # Candle timeframe (1h = hourly)

# Cache Settings
CACHE_DIR = "./data"  # Directory for cached data
FORCE_REFRESH = False  # If True, ignore cache and re-download
CACHE_FORMAT = "parquet"  # Cache file format (parquet or csv)

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Output Directories
OUTPUT_DIR = "./results"
VISUALIZATION_DIR = "./results/best_pumps"

# Visualization Settings
TOP_N_PLOTS = 20  # Number of top anomalies to visualize
CONTEXT_WINDOW_HOURS = 168  # Hours of context around anomaly (7 days)
PLOT_DPI = 150  # Resolution of saved plots

# Deep Black Theme Colors
THEME = {
    "background": "#000000",  # Pure black
    "price_line": "#E0E0E0",  # Off-white
    "ma_line": "#404040",  # Dark gray
    "anomaly_marker": "#FF3333",  # Bright red
    "volume_bar": "#333333",  # Dark grey
    "volume_anomaly": "#FF3333",  # Bright red
    "grid": "#1A1A1A",  # Very subtle grid
    "text": "#FFFFFF",  # White text
    "spine": "#2A2A2A",  # Subtle spines
}

# Plot Size
FIGURE_WIDTH = 16  # Width in inches
FIGURE_HEIGHT = 10  # Height in inches

# Line Widths
PRICE_LINEWIDTH = 0.8  # Main price line
MA_LINEWIDTH = 0.7  # Moving average lines
GRID_LINEWIDTH = 0.3  # Grid lines

# ============================================================================
# LOGGING PARAMETERS
# ============================================================================

# Log Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"

# Log File
LOG_FILE = "./results/backtest.log"

# Console Logging
CONSOLE_LOGGING = True

# ============================================================================
# PERFORMANCE PARAMETERS
# ============================================================================

# Rate Limiting (to avoid Binance API bans)
RATE_LIMIT_ENABLED = True
RETRY_ON_FAILURE = True
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds to wait between retries

# Memory Management
MAX_SYMBOLS_IN_MEMORY = 50  # Process this many symbols at once

# ============================================================================
# FILTER PARAMETERS
# ============================================================================

# Symbol Filters
REQUIRE_USDT_PAIRS = True  # Only analyze USDT pairs for spot
MIN_VOLUME_24H = 1000000  # Minimum 24h volume in USD
EXCLUDE_STABLECOINS = True  # Exclude stablecoin pairs

# Stablecoin symbols to exclude (if EXCLUDE_STABLECOINS = True)
STABLECOIN_SYMBOLS = [
    "USDT",
    "USDC",
    "BUSD",
    "DAI",
    "TUSD",
    "USDP",
    "USDD",
    "GUSD",
    "FRAX",
    "LUSD",
]

# Date Range Filters
EXCLUDE_WEEKENDS = False  # Exclude weekend data
EXCLUDE_HOLIDAYS = False  # Exclude holidays

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Algorithm Variants
USE_MEDIAN_INSTEAD_OF_MEAN = False  # Use median for robustness
USE_EXPONENTIAL_WEIGHTS = False  # Use exponential weights for older data

# Multi-Timeframe Analysis
ENABLE_MULTI_TIMEFRAME = False  # Analyze multiple timeframes
ADDITIONAL_TIMEFRAMES = ["4h", "1d"]  # Additional timeframes to check

# Risk Management
MAX_POSITION_SIZE = 0.05  # Maximum position size (5% of capital)
STOP_LOSS_PCT = 0.10  # Stop loss percentage (10%)
TAKE_PROFIT_PCT = 0.20  # Take profit percentage (20%)

# ============================================================================
# EXPERIMENTAL FEATURES
# ============================================================================

# Machine Learning Enhancement
USE_ML_CLASSIFIER = False  # Use ML to classify pumps
ML_MODEL_PATH = "./models/pump_classifier.pkl"

# Social Media Integration
CHECK_SOCIAL_SIGNALS = False  # Check Twitter/Discord for pump signals
SOCIAL_API_KEY = None

# Real-Time Monitoring
ENABLE_REALTIME = False  # Enable real-time detection
REALTIME_CHECK_INTERVAL = 60  # Seconds between checks

# ============================================================================
# REPORT SETTINGS
# ============================================================================

# Report Format
GENERATE_HTML_REPORT = True  # Generate HTML report
GENERATE_PDF_REPORT = False  # Generate PDF report
GENERATE_JSON_REPORT = True  # Generate JSON report

# Report Contents
INCLUDE_DETAILED_STATS = True  # Include detailed statistics
INCLUDE_SYMBOL_BREAKDOWN = True  # Include per-symbol breakdown
INCLUDE_TIMELINE = True  # Include timeline of events

# ============================================================================
# NOTIFICATION SETTINGS
# ============================================================================

# Notification Channels
ENABLE_EMAIL_NOTIFICATIONS = False
ENABLE_TELEGRAM_NOTIFICATIONS = False
ENABLE_DISCORD_NOTIFICATIONS = False

# Email Settings
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_USERNAME = None
EMAIL_PASSWORD = None
EMAIL_RECIPIENTS = []

# Telegram Settings
TELEGRAM_BOT_TOKEN = None
TELEGRAM_CHAT_ID = None

# Discord Settings
DISCORD_WEBHOOK_URL = None

# ============================================================================
# BACKTESTING SETTINGS
# ============================================================================

# Backtesting Mode
PAPER_TRADING_MODE = False  # Simulate trades without real money
INITIAL_CAPITAL = 10000  # Starting capital in USD

# Transaction Costs
TRADING_FEE = 0.001  # 0.1% trading fee
SLIPPAGE = 0.002  # 0.2% slippage

# Performance Metrics
CALCULATE_SHARPE_RATIO = True  # Calculate Sharpe ratio
CALCULATE_MAX_DRAWDOWN = True  # Calculate maximum drawdown
CALCULATE_WIN_RATE = True  # Calculate win rate

# Benchmark
BENCHMARK_SYMBOL = "BTC/USDT"  # Benchmark for comparison

# ============================================================================
# VALIDATION
# ============================================================================


def validate_config():
    """Validate configuration parameters"""
    errors = []

    # Check window sizes
    if W_SHORT >= W_MEDIUM:
        errors.append("W_SHORT must be less than W_MEDIUM")
    if W_MEDIUM >= W_LONG:
        errors.append("W_MEDIUM must be less than W_LONG")

    # Check thresholds
    if PRICE_THRESHOLD <= 1.0:
        errors.append("PRICE_THRESHOLD must be greater than 1.0")
    if VOLUME_THRESHOLD <= 1.0:
        errors.append("VOLUME_THRESHOLD must be greater than 1.0")

    # Check limits
    if SPOT_LIMIT <= 0 or FUTURES_LIMIT <= 0:
        errors.append("SPOT_LIMIT and FUTURES_LIMIT must be positive")

    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return True


# Validate on import
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"WARNING: {e}")
