"""
Data Ingestion Module — Raw M5 Data
Loads the 3 cleaned CSV files and merges them into a unified DataFrame
for the forecasting pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent.parent
CLEANED_PATH = BASE_PATH / "data_cleaned"

TARGET_STORE = "CA_1"


def load_calendar(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load cleaned calendar data."""
    if filepath is None:
        filepath = str(CLEANED_PATH / "calendar_cleaned.csv")

    logger.info(f"Loading calendar from {filepath}")
    cal = pd.read_csv(filepath, parse_dates=['date'])
    logger.info(f"  Calendar: {len(cal):,} rows, {cal['date'].min().date()} → {cal['date'].max().date()}")
    return cal


def load_sales(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load cleaned sales train validation data."""
    if filepath is None:
        filepath = str(CLEANED_PATH / "sales_train_validation_cleaned.csv")

    logger.info(f"Loading sales from {filepath}")
    sales = pd.read_csv(filepath)
    logger.info(f"  Sales: {len(sales):,} items × {len([c for c in sales.columns if c.startswith('d_')])} days")
    return sales


def load_prices(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load cleaned sell prices data."""
    if filepath is None:
        filepath = str(CLEANED_PATH / "sell_prices_cleaned.csv")

    logger.info(f"Loading prices from {filepath}")
    prices = pd.read_csv(filepath)
    logger.info(f"  Prices: {len(prices):,} records, {prices['item_id'].nunique()} items")
    return prices


def melt_and_merge(sales_df: pd.DataFrame, calendar_df: pd.DataFrame,
                   prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide-format sales into long-format and merge with calendar + prices
    to create a unified DataFrame.

    Output columns: date, sku, store_id, units_sold, price, promo_flag, cat_id, dept_id
    """
    logger.info("Melting and merging data...")

    # Identify columns
    day_cols = [c for c in sales_df.columns if c.startswith('d_')]
    id_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    id_cols = [c for c in id_cols if c in sales_df.columns]

    # Melt wide → long
    long_df = sales_df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name='d',
        value_name='units_sold'
    )
    logger.info(f"  Melted: {len(long_df):,} records")

    # Merge with calendar (get actual dates + promo_flag)
    cal_cols = calendar_df[['d', 'date', 'wm_yr_wk', 'promo_flag']].copy()
    long_df = long_df.merge(cal_cols, on='d', how='left')

    # Merge with prices
    prices_cols = ['item_id', 'wm_yr_wk', 'sell_price']
    if 'store_id' in prices_df.columns:
        prices_filtered = prices_df[prices_df['store_id'] == TARGET_STORE].copy()
    else:
        prices_filtered = prices_df.copy()

    long_df = long_df.merge(
        prices_filtered[prices_cols],
        on=['item_id', 'wm_yr_wk'],
        how='left'
    )

    # Rename to match project schema
    long_df = long_df.rename(columns={
        'item_id': 'sku',
        'sell_price': 'price'
    })

    # Drop rows with missing prices (items not yet on sale)
    before = len(long_df)
    long_df = long_df.dropna(subset=['price'])
    dropped = before - len(long_df)
    if dropped > 0:
        logger.info(f"  Dropped {dropped:,} records with missing prices")

    # Ensure non-negative sales
    long_df['units_sold'] = long_df['units_sold'].clip(lower=0)

    # Select and order final columns
    final_cols = ['date', 'sku', 'store_id', 'units_sold', 'price', 'promo_flag']
    if 'cat_id' in long_df.columns:
        final_cols.append('cat_id')
    if 'dept_id' in long_df.columns:
        final_cols.append('dept_id')

    pos_df = long_df[final_cols].copy()
    pos_df = pos_df.sort_values(['sku', 'date']).reset_index(drop=True)

    logger.info(f"  ✅ Merged data: {len(pos_df):,} records, {pos_df['sku'].nunique()} SKUs")
    logger.info(f"  Date range: {pos_df['date'].min().date()} → {pos_df['date'].max().date()}")

    return pos_df


def load_stock_data(filepath: Optional[str] = None, sku_list: list = None) -> pd.DataFrame:
    """Load current stock levels. Generates default if no file provided."""
    if filepath and Path(filepath).exists():
        return pd.read_csv(filepath)

    if sku_list is None:
        sku_list = []

    default_stock = {
        'sku': sku_list,
        'current_stock': [50] * len(sku_list)
    }
    return pd.DataFrame(default_stock)


def get_sku_list(df: pd.DataFrame) -> list:
    """Get unique list of SKUs."""
    return df['sku'].unique().tolist()


if __name__ == "__main__":
    cal = load_calendar()
    sales = load_sales()
    prices = load_prices()
    merged = melt_and_merge(sales, cal, prices)
    print(merged.head())
    print(f"\nSKUs: {get_sku_list(merged)[:5]}...")
