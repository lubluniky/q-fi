"""
Main Backtesting Script for Pump-and-Dump Detection
Orchestrates data fetching, Rust engine, and visualization
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import quant_engine
except ImportError:
    print("ERROR: quant_engine module not found!")
    print("Please build the Rust module first:")
    print("  cd Pump-and-Dump")
    print("  maturin develop --release")
    sys.exit(1)

from python.data_ingestion import BinanceDataFetcher
from python.visualization import PumpDumpVisualizer, create_summary_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./results/backtest.log"),
    ],
)
logger = logging.getLogger(__name__)


class PumpDumpBacktester:
    """Main backtesting engine for pump-and-dump detection"""

    def __init__(self, cache_dir: str = "./data", output_dir: str = "./results"):
        self.fetcher = BinanceDataFetcher(cache_dir=cache_dir)
        self.visualizer = PumpDumpVisualizer(output_dir=output_dir + "/best_pumps")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_backtest(
        self,
        spot_limit: int = 500,
        futures_limit: int = 500,
        max_workers: int = 10,
        force_refresh: bool = False,
    ) -> List[Tuple[str, pd.DataFrame, List[int], str]]:
        """
        Run the complete backtest pipeline

        Args:
            spot_limit: Number of top spot symbols to analyze
            futures_limit: Number of top futures symbols to analyze
            max_workers: Number of parallel workers for data fetching
            force_refresh: If True, ignore cache and re-fetch all data

        Returns:
            List of (symbol, dataframe, anomaly_indices, market_type) tuples
        """
        logger.info("=" * 80)
        logger.info("PUMP-AND-DUMP DETECTION BACKTEST")
        logger.info("Algorithm: arXiv:2503.08692v1 'Best Setting'")
        logger.info("=" * 80)

        # Step 1: Fetch top symbols
        logger.info("\n[STEP 1] Fetching top liquid symbols from Binance...")
        spot_symbols = self.fetcher.get_top_symbols(
            limit=spot_limit, market_type="spot"
        )
        futures_symbols = self.fetcher.get_top_symbols(
            limit=futures_limit, market_type="futures"
        )

        logger.info(
            f"Selected {len(spot_symbols)} spot + {len(futures_symbols)} futures symbols"
        )

        # Step 2: Download OHLCV data
        logger.info("\n[STEP 2] Downloading OHLCV data (1h candles, 2025-2026)...")
        logger.info("This may take several minutes...")

        start_time = time.time()

        spot_data = self.fetcher.fetch_multiple_parallel(
            spot_symbols,
            market_type="spot",
            max_workers=max_workers,
            force_refresh=force_refresh,
        )

        futures_data = self.fetcher.fetch_multiple_parallel(
            futures_symbols,
            market_type="futures",
            max_workers=max_workers,
            force_refresh=force_refresh,
        )

        download_time = time.time() - start_time
        logger.info(
            f"Downloaded {len(spot_data)} spot + {len(futures_data)} futures datasets "
            f"in {download_time:.1f}s"
        )

        # Step 3: Run anomaly detection
        logger.info("\n[STEP 3] Running Rust-powered anomaly detection...")
        results = []

        all_data = [(s, df, "spot") for s, df in spot_data] + [
            (s, df, "futures") for s, df in futures_data
        ]

        detection_start = time.time()
        processed = 0
        total_anomalies = 0

        for symbol, df, market_type in all_data:
            try:
                # Prepare data for Rust engine
                opens = df["open"].values.tolist()
                highs = df["high"].values.tolist()
                volumes = df["volume"].values.tolist()

                # Call Rust function
                anomaly_indices = quant_engine.detect_anomalies(opens, highs, volumes)

                results.append((symbol, df, anomaly_indices, market_type))
                total_anomalies += len(anomaly_indices)
                processed += 1

                if len(anomaly_indices) > 0:
                    logger.info(
                        f"✓ {symbol} ({market_type}): {len(anomaly_indices)} pump(s) detected"
                    )

            except Exception as e:
                logger.error(f"✗ {symbol} ({market_type}): Detection error - {e}")

        detection_time = time.time() - detection_start
        logger.info(
            f"Processed {processed} symbols in {detection_time:.2f}s "
            f"({processed / detection_time:.1f} symbols/sec)"
        )
        logger.info(f"Total anomalies detected: {total_anomalies}")

        # Step 4: Generate report
        logger.info("\n[STEP 4] Generating summary report...")
        create_summary_report(results, output_dir=str(self.output_dir))

        # Step 5: Visualize top anomalies
        logger.info("\n[STEP 5] Visualizing top 20 pump-and-dump events...")
        self.visualizer.plot_top_anomalies(results, top_n=20)

        # Print final statistics
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total symbols scanned: {processed}")
        logger.info(
            f"Symbols with detected pumps: {sum(1 for _, _, a, _ in results if len(a) > 0)}"
        )
        logger.info(f"Total anomalies detected: {total_anomalies}")
        logger.info(f"Average anomalies per symbol: {total_anomalies / processed:.2f}")
        logger.info(f"Total execution time: {time.time() - start_time:.1f}s")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 80)

        return results

    def run_single_symbol(
        self, symbol: str, market_type: str = "spot"
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Run detection on a single symbol (for testing/debugging)

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            market_type: 'spot' or 'futures'

        Returns:
            Tuple of (dataframe, anomaly_indices)
        """
        logger.info(f"Analyzing {symbol} ({market_type})...")

        # Fetch data
        df = self.fetcher.fetch_and_cache(
            symbol, market_type=market_type, force_refresh=False
        )

        if df is None or df.empty:
            logger.error(f"No data available for {symbol}")
            return None, []

        logger.info(f"Loaded {len(df)} candles")

        # Run detection
        opens = df["open"].values.tolist()
        highs = df["high"].values.tolist()
        volumes = df["volume"].values.tolist()

        anomaly_indices = quant_engine.detect_anomalies(opens, highs, volumes)

        logger.info(f"Detected {len(anomaly_indices)} anomalies")

        # Visualize
        if len(anomaly_indices) > 0:
            self.visualizer.plot_anomaly(
                df, anomaly_indices, symbol, market_type=market_type
            )

        return df, anomaly_indices


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pump-and-Dump Detection Backtest (arXiv:2503.08692v1)"
    )

    parser.add_argument(
        "--spot-limit",
        type=int,
        default=500,
        help="Number of top spot symbols to analyze (default: 500)",
    )

    parser.add_argument(
        "--futures-limit",
        type=int,
        default=500,
        help="Number of top futures symbols to analyze (default: 500)",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel workers for data fetching (default: 10)",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download of all data (ignore cache)",
    )

    parser.add_argument(
        "--single-symbol",
        type=str,
        default=None,
        help="Test on a single symbol (e.g., 'BTC/USDT')",
    )

    parser.add_argument(
        "--market-type",
        type=str,
        default="spot",
        choices=["spot", "futures"],
        help="Market type for single symbol test (default: spot)",
    )

    args = parser.parse_args()

    # Initialize backtester
    backtester = PumpDumpBacktester()

    if args.single_symbol:
        # Single symbol mode (testing)
        logger.info("Running in SINGLE SYMBOL mode")
        backtester.run_single_symbol(args.single_symbol, args.market_type)
    else:
        # Full backtest mode
        logger.info("Running in FULL BACKTEST mode")
        backtester.run_backtest(
            spot_limit=args.spot_limit,
            futures_limit=args.futures_limit,
            max_workers=args.max_workers,
            force_refresh=args.force_refresh,
        )


if __name__ == "__main__":
    main()
