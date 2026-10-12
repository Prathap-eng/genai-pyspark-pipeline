"""Generate synthetic e-commerce datasets and save them as Parquet files."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.config import DEFAULT_SEED, RAW_DATA_DIR, configure_logging
from src.data_generator import SyntheticDataGenerator

LOGGER = logging.getLogger(__name__)


def _format_bytes(size_bytes: int) -> str:
    """Format a file size using a readable binary unit."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def _directory_size(path: Path) -> int:
    """Return the combined size of all files below a file or directory path."""
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def run(
    output_dir: Path = RAW_DATA_DIR,
    customer_count: int = 100_000,
    product_count: int = 10_000,
    order_count: int = 1_000_000,
    seed: int = DEFAULT_SEED,
) -> None:
    """Generate the three datasets and save each as a Parquet dataset."""
    start_time = time.perf_counter()
    generator = SyntheticDataGenerator(customer_count, product_count, order_count, seed)
    customers, products, orders = generator.generate_all()

    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, dataframe in (
        ("customers", customers),
        ("products", products),
        ("orders", orders),
    ):
        output_path = output_dir / f"{dataset_name}.parquet"
        LOGGER.info("Writing %s rows to %s", len(dataframe), output_path)
        dataframe.to_parquet(output_path, index=False)
        size = _directory_size(output_path)
        print(f"{dataset_name}: {len(dataframe):,} rows, {_format_bytes(size)} ({output_path})")

    elapsed = time.perf_counter() - start_time
    print(f"Total generation and save time: {elapsed:.2f} seconds")
    LOGGER.info("All datasets saved successfully in %.2f seconds", elapsed)


def main() -> int:
    """Parse command-line arguments and execute the pipeline with error handling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    configure_logging()

    try:
        run(args.output_dir, args.customers, args.products, args.orders, args.seed)
    except (OSError, ValueError, ImportError) as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1
    except Exception:
        LOGGER.exception("Unexpected error while running the pipeline")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
