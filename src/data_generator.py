"""Generate synthetic e-commerce customers, products, and orders."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm.auto import tqdm

from src.config import DEFAULT_SEED, RAW_DATA_DIR, configure_logging

LOGGER = logging.getLogger(__name__)
CATEGORIES = ("Electronics", "Clothing", "Home", "Sports", "Books")


class SyntheticDataGenerator:
    """Create reproducible synthetic e-commerce data as pandas DataFrames.

    By default, the generator creates 100,000 customers, 10,000 products,
    and 1,000,000 orders. Customer ages follow a normal distribution centered
    around 35. Orders follow a Pareto-style distribution where the top 20% of
    customers account for 80% of orders.
    """

    def __init__(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Initialize dataset sizes and deterministic random number sources."""
        if customer_count < 1 or product_count < 1 or order_count < 0:
            raise ValueError("customer_count and product_count must be positive; order_count cannot be negative")
        self.customer_count = customer_count
        self.product_count = product_count
        self.order_count = order_count
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fake = Faker()
        self.fake.seed_instance(seed)

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer IDs, Faker identity fields, and registration dates."""
        LOGGER.info("Generating %s customers", self.customer_count)
        ages = np.clip(np.rint(self.rng.normal(loc=35, scale=10, size=self.customer_count)), 18, 90).astype(int)
        customers = [
            {
                "customer_id": customer_id,
                "name": self.fake.name(),
                "email": self.fake.unique.email(),
                "age": int(ages[customer_id - 1]),
                "city": self.fake.city(),
                "country": self.fake.country(),
                "registration_date": self.fake.date_between(start_date="-5y", end_date="today"),
            }
            for customer_id in tqdm(range(1, self.customer_count + 1), desc="Customers", unit="customer")
        ]
        frame = pd.DataFrame.from_records(
            customers,
            columns=["customer_id", "name", "email", "age", "city", "country", "registration_date"],
        )
        LOGGER.info("Generated customer DataFrame with %s rows", len(frame))
        return frame

    def generate_products(self) -> pd.DataFrame:
        """Generate products with category, price, stock, and customer rating."""
        LOGGER.info("Generating %s products", self.product_count)
        products = [
            {
                "product_id": product_id,
                "name": f"{self.fake.word().title()} {self.fake.word().title()}",
                "category": str(self.rng.choice(CATEGORIES)),
                "price": round(float(self.rng.uniform(10, 500)), 2),
                "stock": int(self.rng.integers(0, 1_001)),
                "rating": round(float(self.rng.uniform(1, 5)), 1),
            }
            for product_id in tqdm(range(1, self.product_count + 1), desc="Products", unit="product")
        ]
        frame = pd.DataFrame.from_records(
            products, columns=["product_id", "name", "category", "price", "stock", "rating"]
        )
        LOGGER.info("Generated product DataFrame with %s rows", len(frame))
        return frame

    def generate_orders(self) -> pd.DataFrame:
        """Generate orders using a Pareto-style 80/20 customer frequency split."""
        LOGGER.info("Generating %s orders with an 80/20 customer split", self.order_count)
        if self.order_count == 0:
            return pd.DataFrame(
                columns=["order_id", "customer_id", "product_id", "quantity", "order_date"]
            )

        customer_ids = np.arange(1, self.customer_count + 1)
        top_count = max(1, int(np.ceil(self.customer_count * 0.2)))
        top_customers = customer_ids[:top_count]
        other_customers = customer_ids[top_count:]
        top_orders = int(round(self.order_count * 0.8))
        other_orders = self.order_count - top_orders

        # Randomize which customers are in the high-frequency 20% cohort.
        self.rng.shuffle(top_customers)
        if other_customers.size:
            self.rng.shuffle(other_customers)
        if other_customers.size:
            assigned_ids = np.concatenate(
                [
                    self.rng.choice(top_customers, size=top_orders, replace=True),
                    self.rng.choice(other_customers, size=other_orders, replace=True)
                    if other_orders
                    else np.array([], dtype=np.int64),
                ]
            )
        else:
            # With fewer than five customers, a distinct 20% lower cohort may not exist.
            assigned_ids = self.rng.choice(top_customers, size=self.order_count, replace=True)
        self.rng.shuffle(assigned_ids)

        # Build arrays in chunks so tqdm reports progress on million-row runs.
        order_ids = np.arange(1, self.order_count + 1, dtype=np.int64)
        product_ids = np.empty(self.order_count, dtype=np.int64)
        quantities = np.empty(self.order_count, dtype=np.int8)
        date_offsets = np.empty(self.order_count, dtype=np.int32)
        chunk_size = 50_000
        for start in tqdm(range(0, self.order_count, chunk_size), desc="Orders", unit="chunk"):
            end = min(start + chunk_size, self.order_count)
            product_ids[start:end] = self.rng.integers(1, self.product_count + 1, size=end - start)
            quantities[start:end] = self.rng.integers(1, 11, size=end - start)
            date_offsets[start:end] = self.rng.integers(0, 731, size=end - start)

        frame = pd.DataFrame(
            {
                "order_id": order_ids,
                "customer_id": assigned_ids,
                "product_id": product_ids,
                "quantity": quantities,
                "order_date": pd.Timestamp.today().normalize() - pd.to_timedelta(date_offsets, unit="D"),
            }
        )
        LOGGER.info("Generated order DataFrame with %s rows", len(frame))
        return frame

    def generate_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return customer, product, and order DataFrames."""
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders()
        LOGGER.info("Finished generating all synthetic e-commerce datasets")
        return customers, products, orders


def main() -> None:
    """Parse CLI options, generate all datasets, and save them as CSV files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    configure_logging()
    generator = SyntheticDataGenerator(args.customers, args.products, args.orders, args.seed)
    customers, products, orders = generator.generate_all()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in (
        ("customers", customers),
        ("products", products),
        ("orders", orders),
    ):
        output_path = args.output_dir / f"{name}.csv"
        dataframe.to_csv(output_path, index=False)
        LOGGER.info("Saved %s rows to %s", len(dataframe), output_path)


if __name__ == "__main__":
    main()
