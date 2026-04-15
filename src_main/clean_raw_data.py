"""
Raw M5 Data Cleaner
Cleaned the 3 raw M5 data files (calendar.xlsx, sales_train_validation.csv,
sell_prices.csv) and saved cleaned versions to data_cleaned/. //for future reference

Usage:
    python src_main/clean_raw_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from _formatting import TermColors, c, header, info, divider, write_log

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
CLEANED_PATH = BASE_PATH / "data_cleaned"
LOG_PATH = BASE_PATH / "logs1"


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: CLEAN CALENDAR
# ═══════════════════════════════════════════════════════════════════════

def clean_calendar(log_file=None):
    """
    Clean calendar.xlsx → calendar_cleaned.csv
    Parses dates, fills missing event fields, builds promo_flag,
    and validates date continuity.
    """
    header("CLEANING: Calendar Data", "")

    filepath = DATA_PATH / "calendar.xlsx"
    if not filepath.exists():
        filepath = DATA_PATH / "calendar.csv"

    print(f"  Loading calendar from {filepath.name}...")
    write_log(log_file, f"Loading calendar from {filepath.name}")

    cal = pd.read_excel(filepath) if filepath.suffix == '.xlsx' else pd.read_csv(filepath)
    original_shape = cal.shape
    info("Raw shape", f"{original_shape[0]:,} rows × {original_shape[1]} columns")

    # Parse dates
    if 'date' in cal.columns:
        cal['date'] = pd.to_datetime(cal['date'], errors='coerce')
        null_dates = cal['date'].isna().sum()
        if null_dates > 0:
            msg = f"Dropped {null_dates} rows with invalid dates"
            print(f"  ️  {msg}")
            write_log(log_file, msg)
            cal = cal.dropna(subset=['date'])

    # Fill missing event columns
    for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        if col in cal.columns:
            null_count = cal[col].isna().sum()
            cal[col] = cal[col].fillna('')
            if null_count > 0:
                msg = f"Filled {null_count:,} missing values in '{col}'"
                print(f"     {c('→', TermColors.DIM)} {msg}")
                write_log(log_file, msg)

    # Build promo_flag from events + SNAP
    cal['has_event'] = (cal.get('event_name_1', pd.Series([''] * len(cal))) != '').astype(int)
    snap_cols = [col for col in cal.columns if col.startswith('snap_')]
    if snap_cols:
        cal['any_snap'] = cal[snap_cols].max(axis=1)
        cal['promo_flag'] = ((cal['has_event'] == 1) | (cal['any_snap'] == 1)).astype(int)
    else:
        cal['promo_flag'] = cal['has_event']

    # Validate date continuity
    if 'date' in cal.columns:
        cal_sorted = cal.sort_values('date')
        expected = pd.date_range(cal_sorted['date'].min(), cal_sorted['date'].max())
        gaps = len(expected) - len(cal_sorted)
        if gaps > 0:
            msg = f"{gaps} missing dates in calendar"
            print(c(f"  ️  {msg}", TermColors.YELLOW))
            write_log(log_file, msg)
        else:
            print(f"     {c('', TermColors.GREEN)} Date continuity verified — no gaps")

    # Remove duplicates
    dupes = cal.duplicated().sum()
    if dupes > 0:
        cal = cal.drop_duplicates()
        msg = f"Removed {dupes} duplicate rows"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        write_log(log_file, msg)

    # Summary
    print()
    divider()
    info("Cleaned shape", f"{cal.shape[0]:,} rows × {cal.shape[1]} columns")
    if 'date' in cal.columns:
        info("Date range", f"{cal['date'].min().date()} → {cal['date'].max().date()}")
    info("Event days", f"{cal['has_event'].sum():,}")
    info("Promo days", f"{cal['promo_flag'].sum():,}")
    if 'd' in cal.columns:
        info("Day columns", f"{cal['d'].iloc[0]} → {cal['d'].iloc[-1]}")

    out_path = CLEANED_PATH / "calendar_cleaned.csv"
    cal.to_csv(out_path, index=False)
    print(f"\n  {c(' Saved:', TermColors.GREEN)} {out_path.name} ({len(cal):,} rows)")
    write_log(log_file, f"Calendar: {original_shape} → {cal.shape}, saved to {out_path}")

    return cal


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: CLEAN SALES TRAIN VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def clean_sales_train_validation(log_file=None):
    """
    Clean sales_train_validation.csv
    Validates non-negative sales, fills NaNs, removes zero-sales items
    and duplicates.
    """
    header("CLEANING: Sales Train Validation", "")

    filepath = DATA_PATH / "sales_train_validation.csv"
    print(f"  Loading sales data from {filepath.name}...")
    write_log(log_file, f"Loading sales data from {filepath.name}")

    df = pd.read_csv(filepath)
    original_shape = df.shape
    info("Raw shape", f"{original_shape[0]:,} rows × {original_shape[1]:,} columns")

    day_cols = [col for col in df.columns if col.startswith('d_')]
    id_cols = [col for col in df.columns if not col.startswith('d_')]
    info("Day columns", f"{len(day_cols)} (d_1 → d_{len(day_cols)})")
    info("ID columns", f"{', '.join(id_cols)}")

    store_count = df['store_id'].nunique() if 'store_id' in df.columns else 1
    msg = f"Keeping ALL stores: {len(df):,} items across {store_count} stores"
    print(f"     {c('', TermColors.GREEN)} {msg}")
    write_log(log_file, msg)

    # Fix negative sales
    neg_count = (df[day_cols] < 0).sum().sum()
    if neg_count > 0:
        msg = f"Fixed {neg_count:,} negative sales values → 0"
        print(f"     {c('️', TermColors.YELLOW)} {msg}")
        df[day_cols] = df[day_cols].clip(lower=0)
        write_log(log_file, msg)
    else:
        print(f"     {c('', TermColors.GREEN)} No negative sales values found")

    # Fill NaN sales
    null_sales = df[day_cols].isna().sum().sum()
    if null_sales > 0:
        msg = f"Filled {null_sales:,} NaN values in sales columns → 0"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        df[day_cols] = df[day_cols].fillna(0).astype(int)
        write_log(log_file, msg)

    # Remove zero-sales items
    df['total_sales'] = df[day_cols].sum(axis=1)
    zero_items = (df['total_sales'] == 0).sum()
    if zero_items > 0:
        df = df[df['total_sales'] > 0].copy()
        msg = f"Removed {zero_items} items with zero total sales"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        write_log(log_file, msg)

    msg = f"Keeping ALL {len(df):,} items (no sampling)"
    print(f"     {c('', TermColors.GREEN)} {msg}")
    write_log(log_file, msg)

    # Remove duplicates
    id_key = 'item_id' if 'item_id' in df.columns else df.columns[0]
    dupes = df.duplicated(subset=[id_key]).sum()
    if dupes > 0:
        df = df.drop_duplicates(subset=[id_key])
        msg = f"Removed {dupes} duplicate items"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        write_log(log_file, msg)

    df = df.drop(columns=['total_sales'])

    # Summary
    print()
    divider()
    info("Cleaned shape", f"{df.shape[0]:,} rows × {df.shape[1]:,} columns")
    if 'cat_id' in df.columns:
        info("Categories", str(df['cat_id'].value_counts().to_dict()))
    if 'dept_id' in df.columns:
        info("Departments", str(df['dept_id'].value_counts().to_dict()))

    print(f"\n  {c('Top 5 items by sales volume:', TermColors.BOLD)}")
    temp = df.copy()
    temp['total'] = temp[day_cols].sum(axis=1)
    for i, (_, row) in enumerate(temp.nlargest(5, 'total').iterrows(), 1):
        item = row.get('item_id', row.iloc[0])
        cat = row.get('cat_id', 'N/A')
        total = row['total']
        print(f"     {i}. {item:<30} {total:>10,} units  [{cat}]")

    out_path = CLEANED_PATH / "sales_train_validation_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  {c(' Saved:', TermColors.GREEN)} {out_path.name} ({len(df):,} rows)")
    write_log(log_file, f"Sales: {original_shape} → {df.shape}, saved to {out_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: CLEAN SELL PRICES
# ═══════════════════════════════════════════════════════════════════════

def clean_sell_prices(valid_items=None, log_file=None):
    """
    Clean sell_prices.csv
    Filters to valid items, removes null/negative/zero prices,
    drops duplicate records, and validates price ranges.
    """
    header("CLEANING: Sell Prices", "")

    filepath = DATA_PATH / "sell_prices.csv"
    print(f"  Loading prices from {filepath.name}...")
    write_log(log_file, f"Loading prices from {filepath.name}")

    prices = pd.read_csv(filepath)
    original_shape = prices.shape
    info("Raw shape", f"{original_shape[0]:,} rows × {original_shape[1]} columns")

    msg = f"Keeping ALL stores: {len(prices):,} records"
    print(f"     {c('', TermColors.GREEN)} {msg}")
    write_log(log_file, msg)

    # Filter to valid items
    if valid_items is not None and 'item_id' in prices.columns:
        before = len(prices)
        prices = prices[prices['item_id'].isin(valid_items)].copy()
        msg = f"Filtered to {len(valid_items)} valid items: {len(prices):,} records (from {before:,})"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        write_log(log_file, msg)

    # Remove bad prices (null, negative, zero) in one pass
    before = len(prices)
    prices = prices.dropna(subset=['sell_price'])
    prices = prices[prices['sell_price'] > 0].copy()
    removed = before - len(prices)
    if removed > 0:
        msg = f"Removed {removed:,} records with null/zero/negative prices"
        print(f"     {c('️', TermColors.YELLOW)} {msg}")
        write_log(log_file, msg)
    else:
        print(f"     {c('', TermColors.GREEN)} All prices are valid")

    # Remove duplicate records
    dup_cols = [col for col in ['store_id', 'item_id', 'wm_yr_wk'] if col in prices.columns]
    if dup_cols:
        dupes = prices.duplicated(subset=dup_cols).sum()
        if dupes > 0:
            prices = prices.drop_duplicates(subset=dup_cols)
            msg = f"Removed {dupes:,} duplicate price records"
            print(f"     {c('→', TermColors.DIM)} {msg}")
            write_log(log_file, msg)

    # Summary
    print()
    divider()
    info("Cleaned shape", f"{prices.shape[0]:,} rows × {prices.shape[1]} columns")
    if 'item_id' in prices.columns:
        info("Unique items", f"{prices['item_id'].nunique()}")
    info("Price range", f"${prices['sell_price'].min():.2f} – ${prices['sell_price'].max():.2f}")
    info("Mean price", f"${prices['sell_price'].mean():.2f}")
    info("Median price", f"${prices['sell_price'].median():.2f}")

    out_path = CLEANED_PATH / "sell_prices_cleaned.csv"
    prices.to_csv(out_path, index=False)
    print(f"\n  {c(' Saved:', TermColors.GREEN)} {out_path.name} ({len(prices):,} rows)")
    write_log(log_file, f"Prices: {original_shape} → {prices.shape}, saved to {out_path}")

    return prices


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Run all data cleaning steps."""
    import time
    start_time = time.time()

    w = 72
    print()
    print(c("╔" + "═" * (w - 2) + "╗", TermColors.CYAN))
    print(c("║", TermColors.CYAN) + c("  🧹  M5 RAW DATA CLEANING PIPELINE", TermColors.BOLD + TermColors.WHITE).center(w + 11) + c("║", TermColors.CYAN))
    print(c("║", TermColors.CYAN) + c("  Calendar  │  Sales  │  Prices  →  data_cleaned/", TermColors.DIM).center(w + 11) + c("║", TermColors.CYAN))
    print(c("╚" + "═" * (w - 2) + "╝", TermColors.CYAN))

    CLEANED_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    log_path = LOG_PATH / "data_cleaning.log"
    with open(log_path, 'w') as log_file:
        log_file.write(f"M5 Raw Data Cleaning — {datetime.now().isoformat()}\n")
        log_file.write("=" * 60 + "\n\n")

        cal = clean_calendar(log_file)
        sales = clean_sales_train_validation(log_file)

        valid_items = sales['item_id'].unique().tolist() if 'item_id' in sales.columns else None
        prices = clean_sell_prices(valid_items, log_file)

        # Final report
        elapsed = time.time() - start_time
        header("CLEANING COMPLETE", "")
        print(f"\n  {c('Files cleaned:', TermColors.BOLD)}")
        print(f"      calendar_cleaned.csv      {cal.shape[0]:>8,} rows")
        print(f"      sales_cleaned.csv          {sales.shape[0]:>8,} rows")
        print(f"      prices_cleaned.csv         {prices.shape[0]:>8,} rows")
        print(f"\n  {c('Output directory:', TermColors.DIM)}  {CLEANED_PATH}")
        print(f"  {c('Log file:', TermColors.DIM)}          {log_path}")
        print(f"  {c('Time elapsed:', TermColors.DIM)}      {elapsed:.1f}s")
        print()
        print(c("═" * w, TermColors.CYAN))

        log_file.write(f"\nCleaning complete in {elapsed:.1f}s\n")
        log_file.write(f"Calendar:  {cal.shape[0]:,} rows\n")
        log_file.write(f"Sales:     {sales.shape[0]:,} rows\n")
        log_file.write(f"Prices:    {prices.shape[0]:,} rows\n")

    return cal, sales, prices


if __name__ == "__main__":
    main()
