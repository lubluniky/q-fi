"""
Visualization Module for Pump-and-Dump Detection
Deep Black Theme with Minimalist, High-Contrast Design
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Deep Black Theme Configuration
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

# Configure matplotlib for dark theme
plt.style.use("dark_background")


class PumpDumpVisualizer:
    """Visualizer for pump-and-dump anomalies with deep black theme"""

    def __init__(self, output_dir: str = "./results/best_pumps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_anomaly(
        self,
        df: pd.DataFrame,
        anomaly_indices: List[int],
        symbol: str,
        market_type: str = "spot",
        window_hours: int = 168,  # 7 days context
        save: bool = True,
    ) -> Optional[Figure]:
        """
        Plot price and volume with anomaly markers

        Args:
            df: DataFrame with OHLCV data (indexed by timestamp)
            anomaly_indices: List of indices where anomalies occurred
            symbol: Trading pair symbol
            market_type: 'spot' or 'futures'
            window_hours: Hours of context to show around anomaly
            save: If True, save to disk

        Returns:
            Figure object or None
        """
        if not anomaly_indices:
            logger.warning(f"No anomalies to plot for {symbol}")
            return None

        # For each anomaly, create a focused plot
        for anomaly_idx in anomaly_indices[:5]:  # Limit to first 5 per symbol
            try:
                self._plot_single_anomaly(
                    df, anomaly_idx, symbol, market_type, window_hours, save
                )
            except Exception as e:
                logger.error(
                    f"Error plotting anomaly for {symbol} at {anomaly_idx}: {e}"
                )

        return None

    def _plot_single_anomaly(
        self,
        df: pd.DataFrame,
        anomaly_idx: int,
        symbol: str,
        market_type: str,
        window_hours: int,
        save: bool,
    ):
        """Plot a single anomaly with context window"""

        # Calculate window
        start_idx = max(0, anomaly_idx - window_hours // 2)
        end_idx = min(len(df), anomaly_idx + window_hours // 2)

        df_window = df.iloc[start_idx:end_idx].copy()
        anomaly_local_idx = anomaly_idx - start_idx

        if anomaly_local_idx < 0 or anomaly_local_idx >= len(df_window):
            return

        # Create figure with deep black background
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 10), facecolor=THEME["background"], sharex=True
        )

        fig.subplots_adjust(hspace=0.05)

        # --- PRICE SUBPLOT ---
        ax1.set_facecolor(THEME["background"])

        # Plot price line (using close prices)
        ax1.plot(
            df_window.index,
            df_window["close"],
            color=THEME["price_line"],
            linewidth=0.8,
            label="Close Price",
            zorder=2,
        )

        # Plot high prices as thin line
        ax1.plot(
            df_window.index,
            df_window["high"],
            color=THEME["ma_line"],
            linewidth=0.5,
            alpha=0.6,
            linestyle="--",
            label="High",
            zorder=1,
        )

        # Calculate and plot 12h MA
        ma_window = 12
        if len(df_window) >= ma_window:
            df_window["ma_12h"] = df_window["open"].rolling(window=ma_window).mean()
            ax1.plot(
                df_window.index,
                df_window["ma_12h"],
                color=THEME["ma_line"],
                linewidth=0.7,
                linestyle="--",
                label="MA(12h)",
                alpha=0.7,
                zorder=1,
            )

        # Mark the anomaly
        anomaly_time = df_window.index[anomaly_local_idx]
        anomaly_high = df_window.iloc[anomaly_local_idx]["high"]

        ax1.scatter(
            [anomaly_time],
            [anomaly_high],
            color=THEME["anomaly_marker"],
            s=150,
            marker="^",
            edgecolors="white",
            linewidths=1.5,
            label="Pump Detected",
            zorder=5,
        )

        # Vertical line at anomaly
        ax1.axvline(
            anomaly_time,
            color=THEME["anomaly_marker"],
            linewidth=1.0,
            linestyle=":",
            alpha=0.5,
            zorder=1,
        )

        # Styling
        ax1.grid(True, color=THEME["grid"], linewidth=0.3, alpha=0.5)
        ax1.set_ylabel(
            "Price (USDT)", color=THEME["text"], fontsize=11, fontweight="bold"
        )
        ax1.tick_params(colors=THEME["text"], labelsize=9)

        # Spines
        for spine in ax1.spines.values():
            spine.set_edgecolor(THEME["spine"])
            spine.set_linewidth(0.5)

        # Legend
        legend = ax1.legend(
            loc="upper left",
            fontsize=9,
            framealpha=0.3,
            facecolor=THEME["background"],
            edgecolor=THEME["spine"],
        )
        for text in legend.get_texts():
            text.set_color(THEME["text"])

        # --- VOLUME SUBPLOT ---
        ax2.set_facecolor(THEME["background"])

        # Plot volume bars
        colors = [THEME["volume_bar"]] * len(df_window)
        colors[anomaly_local_idx] = THEME["volume_anomaly"]

        ax2.bar(
            df_window.index,
            df_window["volume"],
            width=0.03,
            color=colors,
            edgecolor="none",
            zorder=2,
        )

        # Calculate and plot 12h volume MA
        if len(df_window) >= ma_window:
            df_window["vol_ma_12h"] = (
                df_window["volume"].rolling(window=ma_window).mean()
            )
            ax2.plot(
                df_window.index,
                df_window["vol_ma_12h"],
                color=THEME["ma_line"],
                linewidth=0.7,
                linestyle="--",
                label="MA(12h) Volume",
                alpha=0.7,
                zorder=3,
            )

        # Vertical line at anomaly
        ax2.axvline(
            anomaly_time,
            color=THEME["anomaly_marker"],
            linewidth=1.0,
            linestyle=":",
            alpha=0.5,
            zorder=1,
        )

        # Styling
        ax2.grid(True, color=THEME["grid"], linewidth=0.3, alpha=0.5)
        ax2.set_ylabel("Volume", color=THEME["text"], fontsize=11, fontweight="bold")
        ax2.set_xlabel("Time", color=THEME["text"], fontsize=11, fontweight="bold")
        ax2.tick_params(colors=THEME["text"], labelsize=9)

        # Spines
        for spine in ax2.spines.values():
            spine.set_edgecolor(THEME["spine"])
            spine.set_linewidth(0.5)

        # Legend
        legend = ax2.legend(
            loc="upper left",
            fontsize=9,
            framealpha=0.3,
            facecolor=THEME["background"],
            edgecolor=THEME["spine"],
        )
        for text in legend.get_texts():
            text.set_color(THEME["text"])

        # Calculate metrics
        anomaly_row = df.iloc[anomaly_idx]
        price_spike = (
            anomaly_row["high"]
            / df.iloc[max(0, anomaly_idx - 12) : anomaly_idx]["open"].mean()
            - 1
        ) * 100
        volume_spike = (
            anomaly_row["volume"]
            / df.iloc[max(0, anomaly_idx - 12) : anomaly_idx]["volume"].mean()
            - 1
        ) * 100

        # Title with metrics
        title_text = (
            f"{symbol} ({market_type.upper()}) - Pump-and-Dump Detection\n"
            f"Time: {anomaly_time.strftime('%Y-%m-%d %H:%M')} | "
            f"Price Spike: +{price_spike:.1f}% | Volume Spike: +{volume_spike:.1f}%"
        )

        fig.suptitle(
            title_text,
            color=THEME["text"],
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        # Tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        if save:
            safe_symbol = symbol.replace("/", "_")
            timestamp_str = anomaly_time.strftime("%Y%m%d_%H%M")
            filename = f"{safe_symbol}_{market_type}_{timestamp_str}_pump.png"
            filepath = self.output_dir / filename

            fig.savefig(
                filepath,
                dpi=150,
                facecolor=THEME["background"],
                edgecolor="none",
                bbox_inches="tight",
            )
            logger.info(f"Saved plot: {filename}")

        plt.close(fig)

    def plot_top_anomalies(
        self,
        results: List[Tuple[str, pd.DataFrame, List[int], str]],
        top_n: int = 20,
    ):
        """
        Plot the top N most significant anomalies across all symbols

        Args:
            results: List of (symbol, dataframe, anomaly_indices, market_type) tuples
            top_n: Number of top anomalies to plot
        """
        # Flatten and score anomalies
        scored_anomalies = []

        for symbol, df, anomaly_indices, market_type in results:
            for idx in anomaly_indices:
                try:
                    # Calculate significance score
                    anomaly_row = df.iloc[idx]

                    # Price spike
                    ma_start = max(0, idx - 12)
                    price_ma = df.iloc[ma_start:idx]["open"].mean()
                    price_spike = (
                        (anomaly_row["high"] / price_ma - 1) * 100
                        if price_ma > 0
                        else 0
                    )

                    # Volume spike
                    vol_ma = df.iloc[ma_start:idx]["volume"].mean()
                    volume_spike = (
                        (anomaly_row["volume"] / vol_ma - 1) * 100 if vol_ma > 0 else 0
                    )

                    # Combined score (weighted)
                    score = price_spike * 0.6 + volume_spike * 0.4

                    scored_anomalies.append(
                        {
                            "symbol": symbol,
                            "df": df,
                            "idx": idx,
                            "market_type": market_type,
                            "score": score,
                            "price_spike": price_spike,
                            "volume_spike": volume_spike,
                            "timestamp": df.index[idx],
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error scoring anomaly {symbol}[{idx}]: {e}")

        # Sort by score
        scored_anomalies.sort(key=lambda x: x["score"], reverse=True)

        # Plot top N
        logger.info(f"Plotting top {top_n} anomalies...")
        for i, anomaly in enumerate(scored_anomalies[:top_n], 1):
            logger.info(
                f"[{i}/{top_n}] {anomaly['symbol']} - "
                f"Score: {anomaly['score']:.1f} | "
                f"Price: +{anomaly['price_spike']:.1f}% | "
                f"Volume: +{anomaly['volume_spike']:.1f}%"
            )

            self._plot_single_anomaly(
                anomaly["df"],
                anomaly["idx"],
                anomaly["symbol"],
                anomaly["market_type"],
                window_hours=168,
                save=True,
            )

        logger.info(f"Completed plotting {len(scored_anomalies[:top_n])} anomalies")


def create_summary_report(
    results: List[Tuple[str, pd.DataFrame, List[int], str]],
    output_dir: str = "./results",
):
    """
    Create a summary report of all detected anomalies

    Args:
        results: List of (symbol, dataframe, anomaly_indices, market_type) tuples
        output_dir: Directory to save report
    """
    report_path = Path(output_dir) / "pump_dump_report.txt"

    total_symbols = len(results)
    total_anomalies = sum(len(anomaly_indices) for _, _, anomaly_indices, _ in results)

    symbols_with_pumps = sum(
        1 for _, _, anomaly_indices, _ in results if len(anomaly_indices) > 0
    )

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PUMP-AND-DUMP DETECTION REPORT\n")
        f.write(
            "Based on: Detecting Crypto Pump-and-Dump Schemes (arXiv:2503.08692v1)\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Symbols Scanned: {total_symbols}\n")
        f.write(f"Symbols with Detected Pumps: {symbols_with_pumps}\n")
        f.write(f"Total Anomalies Detected: {total_anomalies}\n")
        f.write(
            f"Average Anomalies per Symbol: {total_anomalies / total_symbols:.2f}\n\n"
        )

        f.write("DETECTED PUMPS BY SYMBOL\n")
        f.write("-" * 80 + "\n")

        for symbol, df, anomaly_indices, market_type in results:
            if len(anomaly_indices) > 0:
                f.write(
                    f"\n{symbol} ({market_type.upper()}) - {len(anomaly_indices)} pump(s)\n"
                )

                for idx in anomaly_indices[:10]:  # Limit to first 10 per symbol
                    try:
                        timestamp = df.index[idx]
                        anomaly_row = df.iloc[idx]

                        ma_start = max(0, idx - 12)
                        price_ma = df.iloc[ma_start:idx]["open"].mean()
                        price_spike = (
                            (anomaly_row["high"] / price_ma - 1) * 100
                            if price_ma > 0
                            else 0
                        )

                        vol_ma = df.iloc[ma_start:idx]["volume"].mean()
                        volume_spike = (
                            (anomaly_row["volume"] / vol_ma - 1) * 100
                            if vol_ma > 0
                            else 0
                        )

                        f.write(
                            f"  - {timestamp.strftime('%Y-%m-%d %H:%M')} | "
                            f"Price: +{price_spike:.1f}% | Volume: +{volume_spike:.1f}%\n"
                        )
                    except Exception as e:
                        f.write(f"  - Error processing index {idx}: {e}\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Summary report saved to {report_path}")


if __name__ == "__main__":
    # Example usage
    visualizer = PumpDumpVisualizer()
    logger.info("Visualizer initialized with deep black theme")
