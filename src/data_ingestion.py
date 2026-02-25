"""
Data Ingestion Module (M5 Adapted)
Handles loading and validation of POS data with store_id support.
No weather/external data for M5 dataset.
"""
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pos_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate POS data from CSV.
    
    Args:
        filepath: Path to the POS CSV file
        
    Returns:
        DataFrame with columns: date, sku, store_id, units_sold, price, promo_flag
    """
    logger.info(f"Loading POS data from {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # Validate required columns
    required_cols = ['date', 'sku', 'units_sold', 'price', 'promo_flag']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Data quality checks
    df = df.dropna(subset=['date', 'sku', 'units_sold'])
    df['units_sold'] = df['units_sold'].clip(lower=0)
    df['promo_flag'] = df['promo_flag'].fillna(0).astype(int)
    
    logger.info(f"Loaded {len(df):,} POS records for {df['sku'].nunique()} SKUs")
    if 'store_id' in df.columns:
        logger.info(f"  Stores: {df['store_id'].unique().tolist()}")
    return df


def load_stock_data(filepath: Optional[str] = None, sku_list: list = None) -> pd.DataFrame:
    """
    Load current stock levels. Generates default if no file provided.
    """
    if filepath and Path(filepath).exists():
        return pd.read_csv(filepath)
    
    if sku_list is None:
        sku_list = []
    
    # Default stock levels based on SKU count
    default_stock = {
        'sku': sku_list,
        'current_stock': [50] * len(sku_list)
    }
    return pd.DataFrame(default_stock)


def get_sku_list(pos_df: pd.DataFrame) -> list:
    """Get unique list of SKUs from POS data."""
    return pos_df['sku'].unique().tolist()


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent / "data"
    pos = load_pos_data(base_path / "pos_data.csv")
    print(pos.head())
    print(f"\nSKUs: {get_sku_list(pos)[:5]}...")
