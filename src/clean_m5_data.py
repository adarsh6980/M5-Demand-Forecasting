"""
M5 Dataset Cleaner & Transformer
Cleans the M5 Forecasting Accuracy raw data and transforms it into
the daily time-series POS format expected by the demand forecasting pipeline.

Input:  M5 raw CSVs (sales_train_evaluation.csv, calendar.csv, sell_prices.csv)
Output: data/pos_data.csv, config/business_rules.yml
"""
import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
CONFIG_PATH = BASE_PATH / "config"

# M5 raw data location
M5_RAW_PATH = Path("/Users/adarsh/Downloads/study/proj/m5-forecasting-accuracy")

# Sampling config
TARGET_STORE = "CA_1"
TOP_N_ITEMS = 30


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & SAMPLE M5 RAW DATA
# ═══════════════════════════════════════════════════════════════════════

def load_and_sample_sales(store: str = TARGET_STORE, top_n: int = TOP_N_ITEMS) -> pd.DataFrame:
    """
    Load sales_train_evaluation.csv and sample top N items from one store.
    
    Steps:
    1. Load wide-format sales data
    2. Filter to target store
    3. Select top N items by total sales volume
    """
    filepath = M5_RAW_PATH / "sales_train_evaluation.csv"
    logger.info(f"Loading M5 sales data from {filepath}...")
    
    df = pd.read_csv(filepath)
    logger.info(f"  Raw: {len(df):,} rows × {len(df.columns):,} columns")
    
    # Filter to target store
    store_df = df[df['store_id'] == store].copy()
    logger.info(f"  Store '{store}': {len(store_df):,} items")
    
    # Identify day columns (d_1, d_2, ..., d_1941)
    day_cols = [c for c in store_df.columns if c.startswith('d_')]
    
    # Calculate total sales per item and pick top N
    store_df['total_sales'] = store_df[day_cols].sum(axis=1)
    top_items = store_df.nlargest(top_n, 'total_sales')
    
    logger.info(f"  Selected top {len(top_items)} items by sales volume")
    logger.info(f"  Categories: {top_items['cat_id'].value_counts().to_dict()}")
    logger.info(f"  Departments: {top_items['dept_id'].value_counts().to_dict()}")
    
    return top_items


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: LOAD CALENDAR & PRICES
# ═══════════════════════════════════════════════════════════════════════

def load_calendar() -> pd.DataFrame:
    """Load and process the M5 calendar with events and SNAP flags."""
    filepath = M5_RAW_PATH / "calendar.csv"
    logger.info(f"Loading calendar from {filepath}...")
    
    cal = pd.read_csv(filepath, parse_dates=['date'])
    
    # Create promo_flag: 1 if any event OR SNAP day for CA
    cal['has_event'] = cal['event_name_1'].notna().astype(int)
    cal['promo_flag'] = ((cal['has_event'] == 1) | (cal['snap_CA'] == 1)).astype(int)
    
    # Build event description for logging
    events = cal[cal['has_event'] == 1]['event_name_1'].value_counts()
    snap_days = cal['snap_CA'].sum()
    
    logger.info(f"  Calendar: {len(cal)} days ({cal['date'].min().date()} → {cal['date'].max().date()})")
    logger.info(f"  Events: {len(events)} unique events, {cal['has_event'].sum()} event days")
    logger.info(f"  SNAP days (CA): {snap_days}")
    logger.info(f"  Total promo days: {cal['promo_flag'].sum()}")
    
    return cal


def load_prices() -> pd.DataFrame:
    """Load sell_prices.csv."""
    filepath = M5_RAW_PATH / "sell_prices.csv"
    logger.info(f"Loading prices from {filepath}...")
    
    prices = pd.read_csv(filepath)
    logger.info(f"  Prices: {len(prices):,} records")
    
    return prices


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: MELT & MERGE TO POS FORMAT
# ═══════════════════════════════════════════════════════════════════════

def transform_to_pos(sales_df: pd.DataFrame, calendar_df: pd.DataFrame,
                     prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform wide-format M5 sales data into long POS format.
    
    Output columns: date, sku, store_id, units_sold, price, promo_flag
    """
    logger.info("Transforming to POS format...")
    
    # Identify day columns
    day_cols = [c for c in sales_df.columns if c.startswith('d_')]
    id_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    # Melt wide → long
    long_df = sales_df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name='d',
        value_name='units_sold'
    )
    logger.info(f"  Melted: {len(long_df):,} records")
    
    # Merge with calendar to get actual dates + promo_flag
    cal_cols = calendar_df[['d', 'date', 'wm_yr_wk', 'promo_flag']].copy()
    long_df = long_df.merge(cal_cols, on='d', how='left')
    
    # Merge with prices
    prices_filtered = prices_df[prices_df['store_id'] == TARGET_STORE].copy()
    long_df = long_df.merge(
        prices_filtered[['item_id', 'wm_yr_wk', 'sell_price']],
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
        logger.info(f"  Dropped {dropped:,} records with missing prices (items not yet on sale)")
    
    # Ensure non-negative sales
    long_df['units_sold'] = long_df['units_sold'].clip(lower=0)
    
    # Select and order final columns
    pos_df = long_df[['date', 'sku', 'store_id', 'units_sold', 'price', 'promo_flag',
                       'cat_id', 'dept_id']].copy()
    pos_df = pos_df.sort_values(['sku', 'date']).reset_index(drop=True)
    
    logger.info(f"  ✅ POS data: {len(pos_df):,} records, {pos_df['sku'].nunique()} SKUs")
    logger.info(f"  Date range: {pos_df['date'].min().date()} → {pos_df['date'].max().date()}")
    logger.info(f"  Price range: ${pos_df['price'].min():.2f} – ${pos_df['price'].max():.2f}")
    logger.info(f"  Avg daily sales: {pos_df['units_sold'].mean():.1f} units")
    
    return pos_df


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE BUSINESS RULES
# ═══════════════════════════════════════════════════════════════════════

def generate_m5_business_rules(pos_df: pd.DataFrame) -> dict:
    """Generate business rules YAML from M5 data, with category-specific logic."""
    logger.info("Generating M5 business rules...")
    
    skus_config = []
    for sku in pos_df['sku'].unique():
        sku_data = pos_df[pos_df['sku'] == sku]
        avg_daily = sku_data['units_sold'].mean()
        max_daily = sku_data['units_sold'].max()
        avg_price = sku_data['price'].mean()
        cat = sku_data['cat_id'].iloc[0]
        
        # Category-specific perishability
        if cat == 'FOODS':
            perishability = np.random.choice([7, 10, 14])  # Short shelf life
            safety_stock = 1
        elif cat == 'HOUSEHOLD':
            perishability = np.random.choice([90, 120, 180])  # Long shelf life
            safety_stock = 3
        else:  # HOBBIES
            perishability = np.random.choice([60, 90, 120])
            safety_stock = 2
        
        config = {
            'sku': sku,
            'category': cat,
            'max_shelf_capacity': int(max(max_daily * 3, avg_daily * 7, 50)),
            'unit_cost': round(float(avg_price * 0.55), 2),  # ~55% COGS for Walmart
            'max_budget_per_order': int(max(500, avg_daily * avg_price * 7)),
            'perishability_days': int(perishability),
            'safety_stock_days': safety_stock
        }
        skus_config.append(config)
    
    return {'skus': skus_config}


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: DATA QUALITY REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_cleaning_report(pos_df: pd.DataFrame, rules: dict):
    """Print a summary report of the cleaned data."""
    w = 62
    print()
    print("═" * w)
    print("  🏪  M5 DATASET — CLEANING & TRANSFORMATION REPORT")
    print("═" * w)
    
    print(f"\n  📊 POS Data (pos_data.csv)")
    print(f"     Records:       {len(pos_df):>10,}")
    print(f"     SKUs:          {pos_df['sku'].nunique():>10}")
    print(f"     Store:         {pos_df['store_id'].iloc[0]:>10}")
    print(f"     Date range:    {pos_df['date'].min().date()} → {pos_df['date'].max().date()}")
    print(f"     Avg daily:     {pos_df['units_sold'].mean():>10.1f} units")
    print(f"     Price range:   ${pos_df['price'].min():.2f} – ${pos_df['price'].max():.2f}")
    print(f"     Promo days:    {pos_df['promo_flag'].sum():>10,} ({pos_df['promo_flag'].mean()*100:.1f}%)")
    
    print(f"\n  📦 Categories:")
    for cat, count in pos_df.groupby('cat_id')['sku'].nunique().items():
        total_sales = pos_df[pos_df['cat_id'] == cat]['units_sold'].sum()
        print(f"     {cat:<15} {count:>3} SKUs │ {total_sales:>10,} total units")
    
    print(f"\n  🏆 Top 5 Products by Sales Volume:")
    top5 = pos_df.groupby('sku')['units_sold'].sum().nlargest(5)
    for i, (sku, total) in enumerate(top5.items(), 1):
        cat = pos_df[pos_df['sku'] == sku]['cat_id'].iloc[0]
        print(f"     {i}. {sku:<30} {total:>8,} units  [{cat}]")
    
    print(f"\n  ⚙️  Business Rules: {len(rules['skus'])} SKUs configured")
    
    print("\n" + "═" * w)
    print("  ✅ Data cleaned and ready for pipeline!")
    print("═" * w)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    w = 62
    print("═" * w)
    print("  🏪  M5 Dataset Cleaner & Transformer")
    print("═" * w)
    
    # Validate raw data exists
    required_files = ['sales_train_evaluation.csv', 'calendar.csv', 'sell_prices.csv']
    for f in required_files:
        if not (M5_RAW_PATH / f).exists():
            logger.error(f"❌ Missing: {M5_RAW_PATH / f}")
            sys.exit(1)
    
    # Step 1: Load & sample sales
    print(f"\n📋 Step 1: Loading & sampling M5 data (store={TARGET_STORE}, top {TOP_N_ITEMS})...")
    sales_df = load_and_sample_sales()
    
    # Step 2: Load calendar & prices
    print("\n📅 Step 2: Loading calendar & prices...")
    calendar_df = load_calendar()
    prices_df = load_prices()
    
    # Step 3: Transform to POS
    print("\n🔄 Step 3: Transforming to POS format...")
    pos_df = transform_to_pos(sales_df, calendar_df, prices_df)
    
    # Step 4: Save POS data
    print("\n💾 Step 4: Saving cleaned data...")
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    pos_df.to_csv(DATA_PATH / 'pos_data.csv', index=False)
    logger.info(f"  Saved pos_data.csv ({len(pos_df):,} rows)")
    
    # Step 5: Generate & save business rules
    print("\n⚙️  Step 5: Generating business rules...")
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    rules = generate_m5_business_rules(pos_df)
    with open(CONFIG_PATH / 'business_rules.yml', 'w') as f:
        yaml.dump(rules, f, default_flow_style=False)
    logger.info(f"  Saved business_rules.yml for {len(rules['skus'])} SKUs")
    
    # Report
    print_cleaning_report(pos_df, rules)
    
    return pos_df, rules


if __name__ == "__main__":
    main()
