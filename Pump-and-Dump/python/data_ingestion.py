"""
Data Ingestion Module for Pump-and-Dump Detection
Fetches 1-hour OHLCV data from Binance (Spot + Futures) with parallel processing
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """Fetches historical OHLCV data from Binance"""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchanges
        self.spot_exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        )

        self.futures_exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )

    def get_top_symbols(self, limit: int = 500, market_type: str = "spot") -> List[str]:
        """
        Get top N symbols by 24h volume

        Args:
            limit: Number of top symbols to return
            market_type: 'spot' or 'futures'

        Returns:
            List of symbol strings
        """
        try:
            exchange = (
                self.spot_exchange if market_type == "spot" else self.futures_exchange
            )

            logger.info(f"Fetching {market_type} tickers from Binance...")
            tickers = exchange.fetch_tickers()

            # Filter and sort by volume
            valid_tickers = []
            for symbol, ticker in tickers.items():
                if ticker.get("quoteVolume") and ticker["quoteVolume"] > 0:
                    # For spot, prefer USDT pairs
                    if market_type == "spot" and not symbol.endswith("/USDT"):
                        continue
                    # For futures, filter out dated contracts
                    if market_type == "futures" and "_" in symbol:
                        continue

                    valid_tickers.append(
                        {"symbol": symbol, "volume": ticker["quoteVolume"]}
                    )

            # Sort by volume descending
            valid_tickers.sort(key=lambda x: x["volume"], reverse=True)

            # Return top N
            top_symbols = [t["symbol"] for t in valid_tickers[:limit]]

            logger.info(f"Selected {len(top_symbols)} {market_type} symbols")
            return top_symbols

        except Exception as e:
            logger.error(f"Error fetching {market_type} symbols: {e}")
            return []

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        market_type: str = "spot",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (default '1h')
            since: Start date
            until: End date
            market_type: 'spot' or 'futures'

        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            exchange = (
                self.spot_exchange if market_type == "spot" else self.futures_exchange
            )

            # Default date range: 2025-01-01 to now
            if since is None:
                since = datetime(2025, 1, 1)
            if until is None:
                until = datetime.now()

            since_ts = int(since.timestamp() * 1000)
            until_ts = int(until.timestamp() * 1000)

            all_candles = []
            current_ts = since_ts

            # Fetch in batches (Binance limit: 1000 candles per request)
            while current_ts < until_ts:
                try:
                    candles = exchange.fetch_ohlcv(
                        symbol, timeframe=timeframe, since=current_ts, limit=1000
                    )

                    if not candles:
                        break

                    all_candles.extend(candles)
                    current_ts = candles[-1][0] + 1

                    # Respect rate limits
                    time.sleep(exchange.rateLimit / 1000)

                    if len(candles) < 1000:
                        break

                except ccxt.NetworkError as e:
                    logger.warning(f"Network error for {symbol}: {e}, retrying...")
                    time.sleep(5)
                    continue
                except ccxt.ExchangeError as e:
                    logger.warning(f"Exchange error for {symbol}: {e}")
                    break

            if not all_candles:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")

            # Filter to requested date range
            df = df[(df.index >= since) & (df.index <= until)]

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} ({market_type}): {e}")
            return None

    def fetch_and_cache(
        self, symbol: str, market_type: str = "spot", force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with caching

        Args:
            symbol: Trading pair symbol
            market_type: 'spot' or 'futures'
            force_refresh: If True, ignore cache and re-fetch

        Returns:
            DataFrame with OHLCV data or None
        """
        # Create cache filename
        safe_symbol = symbol.replace("/", "_")
        cache_file = self.cache_dir / f"{safe_symbol}_{market_type}_1h.parquet"

        # Check cache
        if not force_refresh and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded {symbol} ({market_type}) from cache")
                return df
            except Exception as e:
                logger.warning(f"Cache read error for {symbol}: {e}")

        # Fetch fresh data
        logger.info(f"Fetching {symbol} ({market_type}) from Binance...")
        df = self.fetch_ohlcv(symbol, market_type=market_type)

        if df is not None and not df.empty:
            # Save to cache
            try:
                df.to_parquet(cache_file)
                logger.info(f"Cached {symbol} ({market_type}): {len(df)} candles")
            except Exception as e:
                logger.warning(f"Cache write error for {symbol}: {e}")

        return df

    def fetch_multiple_parallel(
        self,
        symbols: List[str],
        market_type: str = "spot",
        max_workers: int = 10,
        force_refresh: bool = False,
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        Fetch multiple symbols in parallel

        Args:
            symbols: List of symbols to fetch
            market_type: 'spot' or 'futures'
            max_workers: Number of parallel workers
            force_refresh: If True, ignore cache

        Returns:
            List of (symbol, dataframe) tuples
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(
                    self.fetch_and_cache, symbol, market_type, force_refresh
                ): symbol
                for symbol in symbols
            }

            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results.append((symbol, df))
                        logger.info(f"✓ {symbol} ({market_type}): {len(df)} candles")
                    else:
                        logger.warning(f"✗ {symbol} ({market_type}): No data")
                except Exception as e:
                    logger.error(f"✗ {symbol} ({market_type}): {e}")

        return results


def main():
    """Example usage"""
    fetcher = BinanceDataFetcher()

    # Get top symbols
    spot_symbols = fetcher.get_top_symbols(limit=500, market_type="spot")
    futures_symbols = fetcher.get_top_symbols(limit=500, market_type="futures")

    logger.info(
        f"Fetching {len(spot_symbols)} spot + {len(futures_symbols)} futures pairs"
    )

    # Fetch spot data
    spot_data = fetcher.fetch_multiple_parallel(
        spot_symbols, market_type="spot", max_workers=10, force_refresh=False
    )

    # Fetch futures data
    futures_data = fetcher.fetch_multiple_parallel(
        futures_symbols, market_type="futures", max_workers=10, force_refresh=False
    )

    logger.info(
        f"Successfully fetched {len(spot_data)} spot + {len(futures_data)} futures pairs"
    )

    return spot_data, futures_data


if __name__ == "__main__":
    main()
