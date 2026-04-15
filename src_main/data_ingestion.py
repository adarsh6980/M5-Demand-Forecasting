"""
Data Ingestion Module
Loads the 3 cleaned M5 files and merges them into a unified
long-format DataFrame ready for the forecasting pipeline.

Parquet files are used when available (data_cleaned/parquet_cache/)
for a significant speedup over CSVs — especially for prices (194MB → 2MB).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent.parent
CLEANED_PATH = BASE_PATH / "data_cleaned"
PARQUET_PATH = CLEANED_PATH / "parquet_cache"


def _load(csv_name, parquet_name, parse_dates=None):
    """Load from parquet cache if available, otherwise fall back to CSV."""
    pq = PARQUET_PATH / parquet_name
    if pq.exists():
        logger.info(f"Reading {parquet_name} from parquet cache")
        df = pd.read_parquet(pq)
        if parse_dates:
            for col in parse_dates:
                if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col])
        return df
    csv = CLEANED_PATH / csv_name
    logger.info(f"Reading {csv_name} from CSV (parquet cache missing)")
    return pd.read_csv(csv, parse_dates=parse_dates)


def load_calendar(filepath=None):
    """Load cleaned calendar data."""
    if filepath:
        cal = pd.read_csv(filepath, parse_dates=['date'])
    else:
        cal = _load('calendar_cleaned.csv', 'calendar.parquet', parse_dates=['date'])
    logger.info(f"  Calendar: {len(cal):,} rows, {cal['date'].min().date()} → {cal['date'].max().date()}")
    return cal


def load_sales(filepath=None):
    """Load cleaned sales train validation data."""
    if filepath:
        sales = pd.read_csv(filepath)
    else:
        sales = _load('sales_train_validation_cleaned.csv', 'sales.parquet')
    day_count = sum(1 for col in sales.columns if col.startswith('d_'))
    logger.info(f"  Sales: {len(sales):,} items × {day_count} days")
    return sales


def load_prices(filepath=None):
    """Load cleaned sell prices data."""
    if filepath:
        prices = pd.read_csv(filepath)
    else:
        prices = _load('sell_prices_cleaned.csv', 'prices.parquet')
    logger.info(f"  Prices: {len(prices):,} records, {prices['item_id'].nunique()} items")
    return prices


def melt_and_merge(sales_df, calendar_df, prices_df):
    """
    Melt wide-format sales into long-format and merge with calendar + prices
    to produce a unified DataFrame for all stores.

    SKU = store_id + '_' + item_id (unique per store-item combination).
    """
    logger.info("Melting and merging data...")

    day_cols = [col for col in sales_df.columns if col.startswith('d_')]
    id_cols = [col for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
               if col in sales_df.columns]

    # Wide → long
    long_df = sales_df.melt(id_vars=id_cols, value_vars=day_cols,
                            var_name='d', value_name='units_sold')
    logger.info(f"  Melted: {len(long_df):,} records")

    # Join calendar for dates and promo flag
    cal_subset = calendar_df[['d', 'date', 'wm_yr_wk', 'promo_flag']].copy()
    long_df = long_df.merge(cal_subset, on='d', how='left')

    # Join prices on item + week (and store if available)
    price_keys = ['item_id', 'wm_yr_wk']
    if 'store_id' in prices_df.columns and 'store_id' in long_df.columns:
        price_keys.append('store_id')

    long_df = long_df.merge(
        prices_df[price_keys + ['sell_price']],
        on=price_keys, how='left'
    )

    # Build composite SKU identifier
    if 'store_id' in long_df.columns:
        long_df['sku'] = long_df['store_id'] + '_' + long_df['item_id']
    else:
        long_df['sku'] = long_df['item_id']

    long_df = long_df.rename(columns={'sell_price': 'price'})

    # Drop rows where the item wasn't on sale yet (missing price)
    before = len(long_df)
    long_df = long_df.dropna(subset=['price'])
    dropped = before - len(long_df)
    if dropped > 0:
        logger.info(f"  Dropped {dropped:,} records with missing prices")

    long_df['units_sold'] = long_df['units_sold'].clip(lower=0)

    # Select final columns
    keep = ['date', 'sku', 'store_id', 'units_sold', 'price', 'promo_flag']
    for extra in ('cat_id', 'dept_id'):
        if extra in long_df.columns:
            keep.append(extra)

    pos_df = long_df[keep].sort_values(['sku', 'date']).reset_index(drop=True)
    logger.info(f"  Merged data: {len(pos_df):,} records, {pos_df['sku'].nunique()} SKUs")
    logger.info(f"  Date range: {pos_df['date'].min().date()} → {pos_df['date'].max().date()}")
    return pos_df


def load_stock_data(filepath=None, sku_list=None):
    """Load current stock levels, or generate defaults if no file exists."""
    if filepath and Path(filepath).exists():
        return pd.read_csv(filepath)

    sku_list = sku_list or []
    return pd.DataFrame({'sku': sku_list, 'current_stock': [50] * len(sku_list)})


def get_sku_list(df):
    """Return the unique SKU identifiers in the dataset."""
    return df['sku'].unique().tolist()


if __name__ == "__main__":
    cal = load_calendar()
    sales = load_sales()
    prices = load_prices()
    merged = melt_and_merge(sales, cal, prices)
    print(merged.head())
    print(f"\nSKUs: {get_sku_list(merged)[:5]}...")
