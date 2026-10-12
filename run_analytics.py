"""Run the e-commerce PySpark reports and display their results."""

from __future__ import annotations

import logging
import time

from src.config import RAW_DATA_DIR, configure_logging
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load the raw Parquet datasets, display reports, and stop Spark cleanly."""
    configure_logging()
    spark = SalesAnalytics.create_spark_session()

    try:
        datasets = {}
        for dataset_name in ("customers", "products", "orders"):
            start_time = time.perf_counter()
            path = RAW_DATA_DIR / f"{dataset_name}.parquet"
            datasets[dataset_name] = SalesAnalytics.load_parquet(path, spark)
            print(f"Loaded {dataset_name}.parquet in {time.perf_counter() - start_time:.2f} seconds")

        orders = datasets["orders"]
        products = datasets["products"]
        operations = (
            (
                "Top customers by revenue",
                lambda: SalesAnalytics.top_customers_by_revenue(orders, products),
            ),
            ("Sales by category", lambda: SalesAnalytics.sales_by_category(orders, products)),
            ("Monthly trends", lambda: SalesAnalytics.monthly_trends(orders, products)),
        )

        for title, operation in operations:
            start_time = time.perf_counter()
            result = operation()
            print(f"\n{title}:")
            result.show(truncate=False)
            print(f"{title} completed in {time.perf_counter() - start_time:.2f} seconds")
    except Exception:
        LOGGER.exception("Analytics run failed")
        raise
    finally:
        spark.stop()
        LOGGER.info("Spark session stopped")


if __name__ == "__main__":
    main()
