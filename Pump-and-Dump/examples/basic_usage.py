"""
Basic Usage Examples for Pump-and-Dump Detection System

This script demonstrates various ways to use the detection system:
1. Direct Rust API usage
2. Python wrapper usage
3. Custom analysis workflows
4. Result interpretation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Import the Rust engine
try:
    import quant_engine
except ImportError:
    print("ERROR: quant_engine module not found!")
    print("Please build it first: maturin develop --release")
    sys.exit(1)

# Import Python modules
from python.data_ingestion import BinanceDataFetcher
from python.visualization import PumpDumpVisualizer

# ============================================================================
# Example 1: Direct Rust API - Simple Detection
# ============================================================================


def example_1_simple_detection():
    """Most basic usage - detect anomalies in price/volume data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Detection with Synthetic Data")
    print("=" * 80)

    # Create synthetic data (1000 hours = ~42 days)
    n = 1000
    base_price = 100.0
    base_volume = 10000.0

    # Normal market data
    opens = [base_price + np.random.normal(0, 5) for _ in range(n)]
    highs = [o + np.random.uniform(1, 3) for o in opens]
    volumes = [base_volume + np.random.normal(0, 1000) for _ in range(n)]

    # Inject a pump-and-dump at index 800
    pump_idx = 800
    opens[pump_idx] = base_price
    highs[pump_idx] = base_price * 2.5  # 150% spike
    volumes[pump_idx] = base_volume * 8  # 700% spike

    print(f"Created {n} candles with synthetic pump at index {pump_idx}")

    # Run detection
    anomalies = quant_engine.detect_anomalies(opens, highs, volumes)

    print(f"\nResults: {len(anomalies)} anomaly(ies) detected")
    if anomalies:
        print(f"Anomaly indices: {anomalies}")
        if pump_idx in anomalies:
            print(f"✓ Successfully detected the injected pump!")


# ============================================================================
# Example 2: Detection with Metrics
# ============================================================================


def example_2_detection_with_metrics():
    """Get detailed metrics for each detected anomaly"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Detection with Detailed Metrics")
    print("=" * 80)

    # Same synthetic data as Example 1
    n = 1000
    base_price = 100.0
    base_volume = 10000.0

    opens = [base_price + np.random.normal(0, 5) for _ in range(n)]
    highs = [o + np.random.uniform(1, 3) for o in opens]
    volumes = [base_volume + np.random.normal(0, 1000) for _ in range(n)]

    # Inject multiple pumps
    pumps = [800, 850, 900]
    for idx in pumps:
        opens[idx] = base_price
        highs[idx] = base_price * (2.0 + np.random.uniform(0, 1))
        volumes[idx] = base_volume * (6 + np.random.uniform(0, 4))

    # Run detection with metrics
    anomalies = quant_engine.detect_anomalies_with_metrics(opens, highs, volumes)

    print(f"\nResults: {len(anomalies)} anomaly(ies) with metrics")
    print("\nDetailed Breakdown:")
    print(f"{'Index':<8} {'Price Spike':<15} {'Volume Spike':<15}")
    print("-" * 40)

    for idx, price_spike, vol_spike in anomalies:
        print(f"{idx:<8} +{price_spike:<14.1f}% +{vol_spike:<14.1f}%")


# ============================================================================
# Example 3: Calculate Technical Indicators
# ============================================================================


def example_3_technical_indicators():
    """Calculate all technical indicators separately for analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Technical Indicators Calculation")
    print("=" * 80)

    # Create data
    n = 1000
    opens = [100 + np.random.normal(0, 5) for _ in range(n)]
    volumes = [10000 + np.random.normal(0, 1000) for _ in range(n)]

    # Calculate indicators
    ma_open_12h, ma_vol_12h, ewma_vol_20d, max_vol_30d, daily_std = (
        quant_engine.calculate_indicators(opens, volumes)
    )

    print(f"\nCalculated indicators for {n} candles:")
    print(f"  MA(Open, 12h):      {len(ma_open_12h)} values")
    print(f"  MA(Volume, 12h):    {len(ma_vol_12h)} values")
    print(f"  EWMA(Volume, 20d):  {len(ewma_vol_20d)} values")
    print(f"  MAX(Volume, 30d):   {len(max_vol_30d)} values")
    print(f"  Daily Std Dev:      {len(daily_std)} values")

    # Show sample values (skip NaN values)
    print("\nSample values from middle of dataset:")
    idx = n // 2
    print(f"  Index {idx}:")
    print(f"    Open:             {opens[idx]:.2f}")
    print(f"    MA(Open, 12h):    {ma_open_12h[idx]:.2f}")
    print(f"    Volume:           {volumes[idx]:.2f}")
    print(f"    MA(Volume, 12h):  {ma_vol_12h[idx]:.2f}")
    print(f"    EWMA(Volume):     {ewma_vol_20d[idx]:.2f}")


# ============================================================================
# Example 4: Real Data from Binance
# ============================================================================


def example_4_real_data():
    """Fetch real data from Binance and detect pumps"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Real Data from Binance")
    print("=" * 80)

    # Initialize fetcher
    fetcher = BinanceDataFetcher(cache_dir="../data")

    # Fetch data for a single symbol
    print("\nFetching BTC/USDT data...")
    df = fetcher.fetch_and_cache("BTC/USDT", market_type="spot", force_refresh=False)

    if df is None or df.empty:
        print("Failed to fetch data")
        return

    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    # Run detection
    opens = df["open"].values.tolist()
    highs = df["high"].values.tolist()
    volumes = df["volume"].values.tolist()

    anomalies = quant_engine.detect_anomalies_with_metrics(opens, highs, volumes)

    print(f"\nDetected {len(anomalies)} pump-and-dump events")

    if anomalies:
        print("\nTop 5 events:")
        # Sort by price spike
        sorted_anomalies = sorted(anomalies, key=lambda x: x[1], reverse=True)[:5]

        for idx, price_spike, vol_spike in sorted_anomalies:
            timestamp = df.index[idx]
            print(f"  {timestamp}: Price +{price_spike:.1f}%, Volume +{vol_spike:.1f}%")


# ============================================================================
# Example 5: Batch Processing Multiple Symbols
# ============================================================================


def example_5_batch_processing():
    """Process multiple symbols efficiently"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Batch Processing Multiple Symbols")
    print("=" * 80)

    # Initialize fetcher
    fetcher = BinanceDataFetcher(cache_dir="../data")

    # Get top 10 symbols
    symbols = fetcher.get_top_symbols(limit=10, market_type="spot")
    print(f"\nProcessing {len(symbols)} symbols...")

    # Fetch data in parallel
    data = fetcher.fetch_multiple_parallel(
        symbols, market_type="spot", max_workers=5, force_refresh=False
    )

    print(f"\nSuccessfully fetched {len(data)} datasets")

    # Process each symbol
    results = []
    for symbol, df in data:
        opens = df["open"].values.tolist()
        highs = df["high"].values.tolist()
        volumes = df["volume"].values.tolist()

        anomalies = quant_engine.detect_anomalies(opens, highs, volumes)
        results.append((symbol, len(anomalies)))

    # Print results
    print("\nResults:")
    print(f"{'Symbol':<15} {'Pumps Detected':<15}")
    print("-" * 30)
    for symbol, count in results:
        print(f"{symbol:<15} {count:<15}")

    total_pumps = sum(count for _, count in results)
    print(f"\nTotal pumps detected: {total_pumps}")


# ============================================================================
# Example 6: Custom Analysis Workflow
# ============================================================================


def example_6_custom_workflow():
    """Build a custom analysis pipeline"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Custom Analysis Workflow")
    print("=" * 80)

    # Step 1: Generate data
    n = 2000
    base_price = 50.0
    base_volume = 5000.0

    opens = [base_price + np.random.normal(0, 2) for _ in range(n)]
    highs = [o + np.random.uniform(0.5, 2) for o in opens]
    volumes = [base_volume + np.random.normal(0, 500) for _ in range(n)]

    # Inject multiple pumps with varying intensity
    pump_locations = [500, 750, 1000, 1250, 1500]
    pump_intensities = [2.0, 2.5, 3.0, 2.2, 2.8]  # Price multipliers

    for idx, intensity in zip(pump_locations, pump_intensities):
        opens[idx] = base_price
        highs[idx] = base_price * intensity
        volumes[idx] = base_volume * (intensity * 2 + 2)

    print(f"Created {n} candles with {len(pump_locations)} pumps")

    # Step 2: Run detection
    anomalies_with_metrics = quant_engine.detect_anomalies_with_metrics(
        opens, highs, volumes
    )

    print(f"Detected {len(anomalies_with_metrics)} anomalies")

    # Step 3: Custom filtering
    # Filter for high-confidence pumps (price > 100%, volume > 500%)
    high_confidence = [
        (idx, price, vol)
        for idx, price, vol in anomalies_with_metrics
        if price > 100 and vol > 500
    ]

    print(f"High-confidence pumps: {len(high_confidence)}")

    # Step 4: Calculate statistics
    if anomalies_with_metrics:
        price_spikes = [p for _, p, _ in anomalies_with_metrics]
        vol_spikes = [v for _, _, v in anomalies_with_metrics]

        print(f"\nStatistics:")
        print(f"  Average price spike:  +{np.mean(price_spikes):.1f}%")
        print(f"  Average volume spike: +{np.mean(vol_spikes):.1f}%")
        print(f"  Max price spike:      +{np.max(price_spikes):.1f}%")
        print(f"  Max volume spike:     +{np.max(vol_spikes):.1f}%")

    # Step 5: Calculate detection accuracy
    detected_indices = set(idx for idx, _, _ in anomalies_with_metrics)
    injected_indices = set(pump_locations)

    true_positives = detected_indices & injected_indices
    false_positives = detected_indices - injected_indices
    false_negatives = injected_indices - detected_indices

    print(f"\nAccuracy Metrics:")
    print(f"  True Positives:  {len(true_positives)}")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")

    if len(true_positives) + len(false_positives) > 0:
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        print(f"  Precision:       {precision:.1%}")

    if len(true_positives) + len(false_negatives) > 0:
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        print(f"  Recall:          {recall:.1%}")


# ============================================================================
# Example 7: Performance Benchmarking
# ============================================================================


def example_7_performance():
    """Benchmark the Rust engine performance"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Performance Benchmarking")
    print("=" * 80)

    import time

    # Test different data sizes
    sizes = [1000, 5000, 10000, 50000]

    print("\nBenchmarking detection speed:")
    print(f"{'Size':<10} {'Time (ms)':<15} {'Speed (candles/sec)':<25}")
    print("-" * 50)

    for size in sizes:
        # Generate data
        opens = [100 + np.random.normal(0, 5) for _ in range(size)]
        highs = [o + np.random.uniform(1, 3) for o in opens]
        volumes = [10000 + np.random.normal(0, 1000) for _ in range(size)]

        # Benchmark
        start = time.time()
        anomalies = quant_engine.detect_anomalies(opens, highs, volumes)
        elapsed = time.time() - start

        speed = size / elapsed if elapsed > 0 else 0

        print(f"{size:<10} {elapsed * 1000:<15.2f} {speed:<25.0f}")

    print("\n✓ Benchmark complete")


# ============================================================================
# Main Runner
# ============================================================================


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("PUMP-AND-DUMP DETECTION SYSTEM - USAGE EXAMPLES")
    print("=" * 80)

    examples = [
        ("Simple Detection", example_1_simple_detection),
        ("Detection with Metrics", example_2_detection_with_metrics),
        ("Technical Indicators", example_3_technical_indicators),
        ("Real Data from Binance", example_4_real_data),
        ("Batch Processing", example_5_batch_processing),
        ("Custom Workflow", example_6_custom_workflow),
        ("Performance Benchmarking", example_7_performance),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...")

    for name, example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pump-and-Dump Detection Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run specific example (1-7)",
    )

    args = parser.parse_args()

    if args.example:
        examples = {
            1: example_1_simple_detection,
            2: example_2_detection_with_metrics,
            3: example_3_technical_indicators,
            4: example_4_real_data,
            5: example_5_batch_processing,
            6: example_6_custom_workflow,
            7: example_7_performance,
        }
        print(f"\nRunning Example {args.example}...")
        examples[args.example]()
    else:
        main()
