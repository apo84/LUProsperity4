#!/usr/bin/env python3
"""
Plot ROUND1 prices data only.

Per product:
- Bid prices (`bid_price_1..3`) as scatter `x` points
- Ask prices (`ask_price_1..3`) as scatter `+` points
- Trade prices from `trades_round_1_day_*.csv` as red `x` points
- Weighted average price across bid and ask levels
- `mid_price` as a connected series
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
PRICE_FILE_GLOB = "prices_round_1_day_*.csv"
TRADE_FILE_GLOB = "trades_round_1_day_*.csv"

# (day, timestamp) bounds. Adjust as needed.
START_TIME = (0, 990_000)
END_TIME = (0, 999_999)


def map_time(day: int | pd.Series, timestamp: int | pd.Series, min_day: int, max_timestamp: int):
    return (day - min_day) * (max_timestamp + 100) + timestamp


def in_time_window(
    day_series: pd.Series,
    timestamp_series: pd.Series,
    start_time: tuple[int, int],
    end_time: tuple[int, int],
) -> pd.Series:
    start_day, start_ts = int(start_time[0]), int(start_time[1])
    end_day, end_ts = int(end_time[0]), int(end_time[1])
    if (start_day, start_ts) > (end_day, end_ts):
        raise ValueError("START_TIME must be <= END_TIME")

    after_start = (day_series > start_day) | (
        (day_series == start_day) & (timestamp_series >= start_ts)
    )
    before_end = (day_series < end_day) | ((day_series == end_day) & (timestamp_series <= end_ts))
    return after_start & before_end


def load_prices(round_folder: Path) -> pd.DataFrame:
    files = sorted(round_folder.glob(PRICE_FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found matching {PRICE_FILE_GLOB} in {round_folder}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        frame = pd.read_csv(file_path, sep=";")
        frame["source_file"] = file_path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    required_columns = {"day", "timestamp", "product", "mid_price"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    min_day = int(df["day"].min())
    max_timestamp = int(df["timestamp"].max())
    df["time"] = map_time(df["day"], df["timestamp"], min_day, max_timestamp)
    return df


def load_trades(round_folder: Path, min_day: int, max_timestamp: int) -> pd.DataFrame:
    files = sorted(round_folder.glob(TRADE_FILE_GLOB))
    if not files:
        return pd.DataFrame(columns=["day", "timestamp", "product", "trade_price", "quantity", "time"])

    frames: list[pd.DataFrame] = []
    day_pattern = re.compile(r"trades_round_1_day_(-?\d+)\.csv$")
    for file_path in files:
        match = day_pattern.search(file_path.name)
        if not match:
            continue
        day = int(match.group(1))
        frame = pd.read_csv(file_path, sep=";")
        if not {"timestamp", "symbol", "price", "quantity"}.issubset(frame.columns):
            continue
        frame = frame.rename(columns={"symbol": "product", "price": "trade_price"})
        frame["day"] = day
        frame["trade_price"] = pd.to_numeric(frame["trade_price"], errors="coerce")
        frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "product", "trade_price"])
        frames.append(frame[["day", "timestamp", "product", "trade_price", "quantity"]])

    if not frames:
        return pd.DataFrame(columns=["day", "timestamp", "product", "trade_price", "quantity", "time"])

    trades = pd.concat(frames, ignore_index=True)
    trades = trades.sort_values(["day", "timestamp"]).reset_index(drop=True)
    trades["time"] = map_time(trades["day"], trades["timestamp"], min_day, max_timestamp)
    return trades


def build_bid_points(product_df: pd.DataFrame) -> pd.DataFrame:
    points: list[pd.DataFrame] = []
    for level in (1, 2, 3):
        price_col = f"bid_price_{level}"
        volume_col = f"bid_volume_{level}"
        if price_col not in product_df.columns or volume_col not in product_df.columns:
            continue

        level_df = product_df[["time", price_col, volume_col]].copy()
        level_df = level_df.rename(columns={price_col: "bid_price", volume_col: "bid_volume"})
        level_df = level_df.dropna(subset=["bid_price", "bid_volume"])
        level_df["bid_volume"] = pd.to_numeric(level_df["bid_volume"], errors="coerce").abs()
        level_df["bid_price"] = pd.to_numeric(level_df["bid_price"], errors="coerce")
        level_df = level_df.dropna(subset=["bid_price", "bid_volume"])
        points.append(level_df)

    if not points:
        return pd.DataFrame(columns=["time", "bid_price", "bid_volume"])
    return pd.concat(points, ignore_index=True)


def build_ask_points(product_df: pd.DataFrame) -> pd.DataFrame:
    points: list[pd.DataFrame] = []
    for level in (1, 2, 3):
        price_col = f"ask_price_{level}"
        if price_col not in product_df.columns:
            continue

        level_df = product_df[["time", price_col]].copy()
        level_df = level_df.rename(columns={price_col: "ask_price"})
        level_df["ask_price"] = pd.to_numeric(level_df["ask_price"], errors="coerce")
        level_df = level_df.dropna(subset=["ask_price"])
        points.append(level_df)

    if not points:
        return pd.DataFrame(columns=["time", "ask_price"])
    return pd.concat(points, ignore_index=True)


def compute_weighted_avg_book_price(product_df: pd.DataFrame) -> pd.Series:
    numerator = pd.Series(0.0, index=product_df.index)
    denominator = pd.Series(0.0, index=product_df.index)

    for side in ("bid", "ask"):
        for level in (1, 2, 3):
            price_col = f"{side}_price_{level}"
            volume_col = f"{side}_volume_{level}"
            if price_col not in product_df.columns or volume_col not in product_df.columns:
                continue

            prices = pd.to_numeric(product_df[price_col], errors="coerce")
            volumes = pd.to_numeric(product_df[volume_col], errors="coerce").abs()
            valid = prices.notna() & volumes.notna()

            numerator = numerator + (prices.where(valid, 0.0) * volumes.where(valid, 0.0))
            denominator = denominator + volumes.where(valid, 0.0)

    weighted_avg = numerator / denominator
    return weighted_avg.where(denominator > 0)


def plot_product(
    product_df: pd.DataFrame, trades_df: pd.DataFrame, product_name: str, output_path: Path
) -> None:
    bid_points = build_bid_points(product_df)
    ask_points = build_ask_points(product_df)
    weighted_avg_book = compute_weighted_avg_book_price(product_df)
    mid_price = pd.to_numeric(product_df["mid_price"], errors="coerce").where(
        pd.to_numeric(product_df["mid_price"], errors="coerce") > 0
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    if not bid_points.empty:
        ax.scatter(
            bid_points["time"],
            bid_points["bid_price"],
            color="tab:blue",
            marker="x",
            s=28,
            linewidths=1.0,
            label="Bid prices",
        )
    if not ask_points.empty:
        ax.scatter(
            ask_points["time"],
            ask_points["ask_price"],
            color="tab:purple",
            marker="+",
            s=24,
            linewidths=1.0,
            label="Ask prices",
        )
    if not trades_df.empty:
        ax.scatter(
            trades_df["time"],
            trades_df["trade_price"],
            color="red",
            marker="x",
            s=30,
            linewidths=1.0,
            label="Trades",
        )

    ax.plot(
        product_df["time"],
        weighted_avg_book,
        color="tab:green",
        linewidth=1.6,
        marker=".",
        markersize=3,
        label="Weighted avg bid+ask price",
    )
    ax.plot(
        product_df["time"],
        mid_price,
        color="tab:orange",
        linewidth=1.6,
        marker=".",
        markersize=3,
        alpha=0.9,
        label="Mid price",
    )

    ax.set_title(f"{product_name} price time-series (prices data only)")
    ax.set_xlabel("Time (combined day + timestamp)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROUND1 prices-only product charts.")
    parser.add_argument(
        "--round-folder",
        default="ROUND1",
        help="Folder containing prices_round_1_day_*.csv files (default: ROUND1)",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to write chart images (default: plots)",
    )
    args = parser.parse_args()

    round_folder = Path(args.round_folder).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_prices(round_folder)
    min_day = int(df["day"].min())
    max_timestamp = int(df["timestamp"].max())
    trades = load_trades(round_folder, min_day, max_timestamp)
    window_mask = in_time_window(df["day"], df["timestamp"], START_TIME, END_TIME)
    df = df[window_mask].copy()
    if df.empty:
        raise ValueError("No price rows in selected START_TIME/END_TIME window")
    if not trades.empty:
        trade_window_mask = in_time_window(trades["day"], trades["timestamp"], START_TIME, END_TIME)
        trades = trades[trade_window_mask].copy()

    selected_days = sorted(df["day"].unique().tolist())
    print(
        f"Selected window {START_TIME} -> {END_TIME}; "
        f"days={selected_days}; ts={int(df['timestamp'].min())}..{int(df['timestamp'].max())}"
    )

    # Recompute mapped time after filtering keeps axis compact and monotonic.
    df["time"] = map_time(df["day"], df["timestamp"], min_day, max_timestamp)

    for product in PRODUCTS:
        product_df = df[df["product"] == product].copy()
        product_trades = trades[trades["product"] == product].copy() if not trades.empty else trades
        if product_df.empty:
            print(f"Skipping {product}: no rows found.")
            continue
        output_path = output_dir / f"{product.lower()}_prices_only_timeseries.png"
        plot_product(product_df, product_trades, product, output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
