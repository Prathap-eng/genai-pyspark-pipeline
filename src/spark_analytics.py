"""PySpark analytics for synthetic e-commerce order and product data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, configure_logging

LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Provide Spark session setup, input loading, and sales analytics."""

    @staticmethod
    def create_spark_session(app_name: str = "EcommerceSalesAnalytics") -> SparkSession:
        """Create a local Spark session with 4 GB memory and performance settings."""
        return (
            SparkSession.builder.appName(app_name)
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()
        )

    @staticmethod
    def load_parquet(path: str | Path, spark: SparkSession | None = None) -> DataFrame:
        """Load a Parquet path into a Spark DataFrame.

        Args:
            path: File or directory containing Parquet data.
            spark: Active Spark session. Uses the active session if omitted.
        """
        session = spark or SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError("No active Spark session; create one before loading Parquet data")
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Parquet path does not exist: {source}")
        LOGGER.info("Loading Parquet data from %s", source)
        return session.read.parquet(str(source))

    @staticmethod
    def top_customers_by_revenue(
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the top N customers ranked by total order revenue."""
        if n < 1:
            raise ValueError("n must be at least 1")
        return (
            orders_df.join(products_df.select("product_id", "price"), on="product_id", how="inner")
            .withColumn("revenue", F.col("quantity") * F.col("price"))
            .groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("total_spend"))
            .orderBy(F.desc("total_spend"))
            .limit(n)
        )

    @staticmethod
    def sales_by_category(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return revenue and units sold grouped by product category."""
        return (
            orders_df.join(
                products_df.select("product_id", "category", "price"),
                on="product_id",
                how="inner",
            )
            .withColumn("revenue", F.col("quantity") * F.col("price"))
            .groupBy("category")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    @staticmethod
    def monthly_trends(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate monthly revenue and month-over-month revenue growth percent.

        The first month has a null growth value because it has no previous month
        for comparison. Growth is also null if the prior month's revenue is zero.
        """
        monthly_revenue = (
            orders_df.join(products_df.select("product_id", "price"), on="product_id", how="inner")
            .withColumn("revenue", F.col("quantity") * F.col("price"))
            .withColumn("month", F.date_trunc("month", F.to_date("order_date")))
            .groupBy("month")
            .agg(F.sum("revenue").alias("monthly_revenue"))
        )
        month_window = Window.orderBy("month")
        with_previous = monthly_revenue.withColumn(
            "previous_month_revenue", F.lag("monthly_revenue").over(month_window)
        )
        return with_previous.withColumn(
            "revenue_growth_pct",
            F.when(
                F.col("previous_month_revenue").isNotNull()
                & (F.col("previous_month_revenue") != 0),
                F.round(
                    (F.col("monthly_revenue") - F.col("previous_month_revenue"))
                    / F.col("previous_month_revenue")
                    * 100,
                    2,
                ),
            ),
        ).orderBy("month")


def run_analytics(raw_dir: Path = RAW_DATA_DIR, output_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Load generated Parquet inputs, compute reports, and save them as Parquet."""
    spark = SalesAnalytics.create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    try:
        orders = SalesAnalytics.load_parquet(raw_dir / "orders.parquet", spark)
        products = SalesAnalytics.load_parquet(raw_dir / "products.parquet", spark)
        reports = {
            "top_customers_by_revenue": SalesAnalytics.top_customers_by_revenue(orders, products),
            "sales_by_category": SalesAnalytics.sales_by_category(orders, products),
            "monthly_trends": SalesAnalytics.monthly_trends(orders, products),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, report in reports.items():
            destination = output_dir / name
            LOGGER.info("Writing report %s to %s", name, destination)
            report.write.mode("overwrite").parquet(str(destination))
        LOGGER.info("Analytics completed; reports saved under %s", output_dir)
    finally:
        spark.stop()


def main() -> None:
    """Parse command-line options and run the Spark analytics job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()
    configure_logging()
    try:
        run_analytics(args.raw_dir, args.output_dir)
    except Exception:
        LOGGER.exception("Spark analytics failed")
        raise


if __name__ == "__main__":
    main()
