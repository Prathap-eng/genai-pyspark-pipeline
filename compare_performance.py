"""Compare pandas and PySpark on the same Parquet sales workload."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import RAW_DATA_DIR, configure_logging
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)


def _timed(operation: Callable[[], object]) -> tuple[object, float]:
    """Run an operation and return its result and elapsed wall-clock seconds."""
    start = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - start


def _print_comparison(results: dict[str, dict[str, float]]) -> None:
    """Print the measured pandas and Spark durations in a simple table."""
    headers = ("Operation", "Pandas (s)", "PySpark (s)")
    rows = [
        (name, f"{times['pandas']:.3f}", f"{times['pyspark']:.3f}")
        for name, times in results.items()
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("\nPerformance comparison (seconds)")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def compare(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Time loading, joining, revenue aggregation, and top-10 selection."""
    orders_path = raw_dir / "orders.parquet"
    products_path = raw_dir / "products.parquet"
    for parquet_path in (orders_path, products_path):
        if not parquet_path.exists():
            raise FileNotFoundError(f"Required Parquet input not found: {parquet_path}")

    results: dict[str, dict[str, float]] = {}

    pandas_start = time.perf_counter()
    pandas_orders = pd.read_parquet(orders_path)
    pandas_products = pd.read_parquet(products_path)
    results["Load Parquet"] = {"pandas": time.perf_counter() - pandas_start}
    LOGGER.info(
        "Pandas loaded %,d orders and %,d products",
        len(pandas_orders),
        len(pandas_products),
    )

    spark = SalesAnalytics.create_spark_session("PandasVsPySparkPerformance")
    spark.sparkContext.setLogLevel("WARN")
    try:
        spark_start = time.perf_counter()
        spark_orders = spark.read.parquet(str(orders_path)).persist(StorageLevel.MEMORY_AND_DISK)
        spark_products = spark.read.parquet(str(products_path)).persist(StorageLevel.MEMORY_AND_DISK)
        spark_order_count = spark_orders.count()
        spark_product_count = spark_products.count()
        results["Load Parquet"]["pyspark"] = time.perf_counter() - spark_start
        LOGGER.info("PySpark loaded %,d orders and %,d products", spark_order_count, spark_product_count)

        pandas_start = time.perf_counter()
        pandas_joined = pandas_orders.merge(
                pandas_products[["product_id", "price"]],
                on="product_id",
                how="inner",
                validate="many_to_one",
        )
        pandas_joined["revenue"] = pandas_joined["quantity"] * pandas_joined["price"]
        results["Join and calculate revenue"] = {"pandas": time.perf_counter() - pandas_start}

        spark_joined = spark_orders.join(
            spark_products.select("product_id", "price"), on="product_id", how="inner"
        ).withColumn("revenue", F.col("quantity") * F.col("price"))
        spark_joined = spark_joined.persist(StorageLevel.MEMORY_AND_DISK)
        _, results["Join and calculate revenue"]["pyspark"] = _timed(spark_joined.count)

        pandas_totals, elapsed = _timed(
            lambda: pandas_joined.groupby("customer_id", as_index=False)["revenue"].sum()
        )
        results["Group revenue by customer"] = {"pandas": elapsed}

        spark_totals = spark_joined.groupBy("customer_id").agg(F.sum("revenue").alias("total_revenue"))
        spark_totals = spark_totals.persist(StorageLevel.MEMORY_AND_DISK)
        _, results["Group revenue by customer"]["pyspark"] = _timed(spark_totals.count)

        pandas_top_ten, elapsed = _timed(
            lambda: pandas_totals.nlargest(10, "revenue").reset_index(drop=True)
        )
        results["Select top 10"] = {"pandas": elapsed}
        spark_top_ten: DataFrame = spark_totals.orderBy(F.desc("total_revenue")).limit(10)
        spark_top_rows, elapsed = _timed(spark_top_ten.collect)
        results["Select top 10"]["pyspark"] = elapsed

        print(f"Input rows: {len(pandas_orders):,} orders, {len(pandas_products):,} products")
        print(f"Top 10 results: Pandas={len(pandas_top_ten)}, PySpark={len(spark_top_rows)}")
        _print_comparison(results)
    finally:
        spark.stop()


def main() -> None:
    """Parse the input directory and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()
    configure_logging()
    try:
        compare(args.raw_dir)
    except Exception:
        LOGGER.exception("Performance comparison failed")
        raise


if __name__ == "__main__":
    main()
