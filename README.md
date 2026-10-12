# E-commerce Data Pipeline

A small Python project that generates reproducible synthetic customer, product, and order data, then uses PySpark to calculate useful business summaries.

## Project layout

```text
genai-pyspark-pipeline/
├── data/
│   ├── processed/          # Spark analysis outputs (created when the pipeline runs)
│   └── raw/                # Generated CSV inputs (created when the generator runs)
├── notebooks/              # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/                  # Test scripts
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10 or newer
- Java 11 or newer for PySpark (the Java runtime must be available on `PATH`)

Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

From the project root, generate 1,000 customers, 100 products, and 5,000 orders:

```bash
python -m src.data_generator
python -m src.spark_analytics
```

The generator writes `customers.csv`, `products.csv`, and `orders.csv` to `data/raw/`. The Spark job reads them and writes Parquet summaries into `data/processed/`:

- `sales_by_category/`
- `monthly_sales/`
- `top_products/`
- `customer_order_summary/`

Run either module with `--help` for its command-line options. Generation is deterministic for a given seed. Spark output directories are replaced each time the analytics job runs.

## Data model and metrics

- **Customers:** ID, name, email, city, country, and signup date.
- **Products:** ID, name, category, price, and stock quantity.
- **Orders:** ID, customer and product IDs, order date, quantity, unit price, and status.
- **Sales by category:** order count, units, and revenue for delivered orders.
- **Monthly sales:** order count and revenue for delivered orders.
- **Top products:** units and revenue for delivered orders.
- **Customer summary:** order count, delivered revenue, and most recent order date.

All amounts are synthetic USD values. Revenue summaries include only delivered orders; customer order counts and latest dates include all statuses.

## Logging

Both modules emit useful progress and error details to the console. Set `LOG_LEVEL=DEBUG` to show debug messages, for example:

```bash
# Windows PowerShell
$env:LOG_LEVEL = "DEBUG"
python -m src.data_generator
```
