"""
Raw M5 Data Cleaner
Cleans the 3 raw M5 data files (calendar.xlsx, sales_train_validation.csv,
sell_prices.csv) and saves cleaned versions to data_cleaned/.

Usage:
    python src_main/clean_raw_data.py
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
CLEANED_PATH = BASE_PATH / "data_cleaned"
LOG_PATH = BASE_PATH / "logs1"

# Config — no store filter, use ALL stores and ALL items
# TARGET_STORE removed to include all stores


# ═══════════════════════════════════════════════════════════════════════
# TERMINAL FORMATTING
# ═══════════════════════════════════════════════════════════════════════

class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def c(text, color):
    return f"{color}{text}{TermColors.RESET}"


def header(title, emoji="", width=72):
    print()
    print(c("═" * width, TermColors.CYAN))
    if emoji:
        print(c(f"  {emoji}  {title}", TermColors.BOLD + TermColors.WHITE))
    else:
        print(c(f"  {title}", TermColors.BOLD + TermColors.WHITE))
    print(c("═" * width, TermColors.CYAN))


def info(label, value, indent=5):
    spaces = " " * indent
    print(f"{spaces}{c(label + ':', TermColors.DIM)}  {value}")


def divider(char="─", width=72):
    print(c(f"  {char * (width - 4)}", TermColors.DIM))


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: CLEAN CALENDAR
# ═══════════════════════════════════════════════════════════════════════

def clean_calendar(log_file=None):
    """
    Clean calendar.xlsx → calendar_cleaned.csv
    - Convert Excel to CSV
    - Parse dates properly
    - Handle missing event fields
    - Create promo_flag from events + SNAP
    - Validate date continuity
    """
    header("CLEANING: Calendar Data", "📅")

    filepath = DATA_PATH / "calendar.xlsx"
    if not filepath.exists():
        # Try CSV fallback
        filepath = DATA_PATH / "calendar.csv"

    msg = f"  Loading calendar from {filepath.name}..."
    print(msg)
    if log_file:
        log_file.write(msg + "\n")

    if filepath.suffix == '.xlsx':
        cal = pd.read_excel(filepath)
    else:
        cal = pd.read_csv(filepath)

    original_shape = cal.shape
    info("Raw shape", f"{original_shape[0]:,} rows × {original_shape[1]} columns")

    # ── Parse dates ──
    if 'date' in cal.columns:
        cal['date'] = pd.to_datetime(cal['date'], errors='coerce')
        null_dates = cal['date'].isna().sum()
        if null_dates > 0:
            msg = f"  ⚠️  Dropped {null_dates} rows with invalid dates"
            print(c(msg, TermColors.YELLOW))
            if log_file:
                log_file.write(msg + "\n")
            cal = cal.dropna(subset=['date'])

    # ── Handle missing event columns ──
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in event_cols:
        if col in cal.columns:
            null_count = cal[col].isna().sum()
            cal[col] = cal[col].fillna('')
            if null_count > 0:
                msg = f"  Filled {null_count:,} missing values in '{col}'"
                print(f"     {c('→', TermColors.DIM)} {msg}")
                if log_file:
                    log_file.write(msg + "\n")

    # ── Create promo_flag ──
    cal['has_event'] = (cal.get('event_name_1', pd.Series([''] * len(cal))) != '').astype(int)

    # Create promo_flag from events + any SNAP
    snap_cols = [col for col in cal.columns if col.startswith('snap_')]
    if snap_cols:
        cal['any_snap'] = cal[snap_cols].max(axis=1)
        cal['promo_flag'] = ((cal['has_event'] == 1) | (cal['any_snap'] == 1)).astype(int)
    else:
        cal['promo_flag'] = cal['has_event']

    # ── Validate date continuity ──
    if 'date' in cal.columns:
        cal_sorted = cal.sort_values('date')
        date_range = pd.date_range(cal_sorted['date'].min(), cal_sorted['date'].max())
        missing_dates = len(date_range) - len(cal_sorted)
        if missing_dates > 0:
            msg = f"  ⚠️  {missing_dates} missing dates in calendar"
            print(c(msg, TermColors.YELLOW))
            if log_file:
                log_file.write(msg + "\n")
        else:
            print(f"     {c('✅', TermColors.GREEN)} Date continuity verified — no gaps")

    # ── Remove duplicate rows ──
    dupes = cal.duplicated().sum()
    if dupes > 0:
        cal = cal.drop_duplicates()
        msg = f"  Removed {dupes} duplicate rows"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        if log_file:
            log_file.write(msg + "\n")

    # ── Summary ──
    print()
    divider()
    info("Cleaned shape", f"{cal.shape[0]:,} rows × {cal.shape[1]} columns")
    if 'date' in cal.columns:
        info("Date range", f"{cal['date'].min().date()} → {cal['date'].max().date()}")
    info("Event days", f"{cal['has_event'].sum():,}")
    info("Promo days", f"{cal['promo_flag'].sum():,}")
    if 'd' in cal.columns:
        info("Day columns", f"{cal['d'].iloc[0]} → {cal['d'].iloc[-1]}")

    # Save
    out_path = CLEANED_PATH / "calendar_cleaned.csv"
    cal.to_csv(out_path, index=False)
    print(f"\n  {c('💾 Saved:', TermColors.GREEN)} {out_path.name} ({len(cal):,} rows)")

    if log_file:
        log_file.write(f"Calendar: {original_shape} → {cal.shape}, saved to {out_path}\n")

    return cal


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: CLEAN SALES TRAIN VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def clean_sales_train_validation(log_file=None):
    """
    Clean sales_train_validation.csv
    - Filter to target store (CA_1)
    - Keep ALL items (no sampling)
    - Validate non-negative sales
    - Check for missing day columns
    - Remove items with zero total sales
    """
    header("CLEANING: Sales Train Validation", "🛒")

    filepath = DATA_PATH / "sales_train_validation.csv"
    msg = f"  Loading sales data from {filepath.name}..."
    print(msg)
    if log_file:
        log_file.write(msg + "\n")

    df = pd.read_csv(filepath)
    original_shape = df.shape
    info("Raw shape", f"{original_shape[0]:,} rows × {original_shape[1]:,} columns")

    # ── Identify day columns ──
    day_cols = [col for col in df.columns if col.startswith('d_')]
    id_cols = [col for col in df.columns if not col.startswith('d_')]
    info("Day columns", f"{len(day_cols)} (d_1 → d_{len(day_cols)})")
    info("ID columns", f"{', '.join(id_cols)}")

    # ── Keep ALL stores (no store filter) ──
    store_df = df.copy()
    msg = f"  Keeping ALL stores: {len(store_df):,} items across {df['store_id'].nunique() if 'store_id' in df.columns else 1} stores"
    print(f"     {c('✅', TermColors.GREEN)} {msg}")
    if log_file:
        log_file.write(msg + "\n")

    # ── Fix negative sales ──
    neg_count = (store_df[day_cols] < 0).sum().sum()
    if neg_count > 0:
        msg = f"  Fixed {neg_count:,} negative sales values → 0"
        print(f"     {c('⚠️', TermColors.YELLOW)} {msg}")
        store_df[day_cols] = store_df[day_cols].clip(lower=0)
        if log_file:
            log_file.write(msg + "\n")
    else:
        print(f"     {c('✅', TermColors.GREEN)} No negative sales values found")

    # ── Fill NaN values in sales ──
    null_sales = store_df[day_cols].isna().sum().sum()
    if null_sales > 0:
        msg = f"  Filled {null_sales:,} NaN values in sales columns → 0"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        store_df[day_cols] = store_df[day_cols].fillna(0).astype(int)
        if log_file:
            log_file.write(msg + "\n")

    # ── Remove zero-sales items ──
    store_df['total_sales'] = store_df[day_cols].sum(axis=1)
    zero_items = (store_df['total_sales'] == 0).sum()
    if zero_items > 0:
        store_df = store_df[store_df['total_sales'] > 0].copy()
        msg = f"  Removed {zero_items} items with zero total sales"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        if log_file:
            log_file.write(msg + "\n")

    # ── Keep ALL items (no top-N sampling) ──
    top_items = store_df.copy()
    msg = f"  Keeping ALL {len(top_items):,} items (no sampling)"
    print(f"     {c('✅', TermColors.GREEN)} {msg}")
    if log_file:
        log_file.write(msg + "\n")

    # ── Remove duplicates ──
    id_key = 'item_id' if 'item_id' in top_items.columns else top_items.columns[0]
    dupes = top_items.duplicated(subset=[id_key]).sum()
    if dupes > 0:
        top_items = top_items.drop_duplicates(subset=[id_key])
        msg = f"  Removed {dupes} duplicate items"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        if log_file:
            log_file.write(msg + "\n")

    # ── Drop helper column ──
    top_items = top_items.drop(columns=['total_sales'])

    # ── Summary ──
    print()
    divider()
    info("Cleaned shape", f"{top_items.shape[0]:,} rows × {top_items.shape[1]:,} columns")
    if 'cat_id' in top_items.columns:
        info("Categories", str(top_items['cat_id'].value_counts().to_dict()))
    if 'dept_id' in top_items.columns:
        info("Departments", str(top_items['dept_id'].value_counts().to_dict()))

    # Show top 5 items
    print(f"\n  {c('Top 5 items by sales volume:', TermColors.BOLD)}")
    temp = top_items.copy()
    temp['total'] = temp[day_cols].sum(axis=1)
    for i, (_, row) in enumerate(temp.nlargest(5, 'total').iterrows(), 1):
        item = row.get('item_id', row.iloc[0])
        cat = row.get('cat_id', 'N/A')
        total = row['total']
        print(f"     {i}. {item:<30} {total:>10,} units  [{cat}]")

    # Save
    out_path = CLEANED_PATH / "sales_train_validation_cleaned.csv"
    top_items.to_csv(out_path, index=False)
    print(f"\n  {c('💾 Saved:', TermColors.GREEN)} {out_path.name} ({len(top_items):,} rows)")

    if log_file:
        log_file.write(f"Sales: {original_shape} → {top_items.shape}, saved to {out_path}\n")

    return top_items


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: CLEAN SELL PRICES
# ═══════════════════════════════════════════════════════════════════════

def clean_sell_prices(valid_items=None, log_file=None):
    """
    Clean sell_prices.csv
    - Filter to target store
    - Filter to valid items (from sales data)
    - Remove null and negative prices
    - Remove duplicate price records
    - Validate price ranges
    """
    header("CLEANING: Sell Prices", "💰")

    filepath = DATA_PATH / "sell_prices.csv"
    msg = f"  Loading prices from {filepath.name}..."
    print(msg)
    if log_file:
        log_file.write(msg + "\n")

    prices = pd.read_csv(filepath)
    original_shape = prices.shape
    info("Raw shape", f"{original_shape[0]:,} rows × {original_shape[1]} columns")

    # ── Keep ALL stores (no store filter) ──
    msg = f"  Keeping ALL stores: {len(prices):,} records"
    print(f"     {c('✅', TermColors.GREEN)} {msg}")
    if log_file:
        log_file.write(msg + "\n")

    # ── Filter to valid items ──
    if valid_items is not None and 'item_id' in prices.columns:
        before = len(prices)
        prices = prices[prices['item_id'].isin(valid_items)].copy()
        msg = f"  Filtered to {len(valid_items)} valid items: {len(prices):,} records (from {before:,})"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        if log_file:
            log_file.write(msg + "\n")

    # ── Remove null prices ──
    null_prices = prices['sell_price'].isna().sum()
    if null_prices > 0:
        prices = prices.dropna(subset=['sell_price'])
        msg = f"  Dropped {null_prices:,} records with null prices"
        print(f"     {c('⚠️', TermColors.YELLOW)} {msg}")
        if log_file:
            log_file.write(msg + "\n")
    else:
        print(f"     {c('✅', TermColors.GREEN)} No null prices found")

    # ── Remove negative prices ──
    neg_prices = (prices['sell_price'] < 0).sum()
    if neg_prices > 0:
        prices = prices[prices['sell_price'] >= 0].copy()
        msg = f"  Removed {neg_prices:,} records with negative prices"
        print(f"     {c('⚠️', TermColors.YELLOW)} {msg}")
        if log_file:
            log_file.write(msg + "\n")
    else:
        print(f"     {c('✅', TermColors.GREEN)} No negative prices found")

    # ── Remove zero prices ──
    zero_prices = (prices['sell_price'] == 0).sum()
    if zero_prices > 0:
        prices = prices[prices['sell_price'] > 0].copy()
        msg = f"  Removed {zero_prices:,} records with zero prices"
        print(f"     {c('→', TermColors.DIM)} {msg}")
        if log_file:
            log_file.write(msg + "\n")

    # ── Remove duplicates ──
    dup_cols = ['store_id', 'item_id', 'wm_yr_wk'] if 'store_id' in prices.columns else ['item_id', 'wm_yr_wk']
    available_dup_cols = [col for col in dup_cols if col in prices.columns]
    if available_dup_cols:
        dupes = prices.duplicated(subset=available_dup_cols).sum()
        if dupes > 0:
            prices = prices.drop_duplicates(subset=available_dup_cols)
            msg = f"  Removed {dupes:,} duplicate price records"
            print(f"     {c('→', TermColors.DIM)} {msg}")
            if log_file:
                log_file.write(msg + "\n")

    # ── Validate price ranges ──
    price_stats = prices['sell_price'].describe()

    # ── Summary ──
    print()
    divider()
    info("Cleaned shape", f"{prices.shape[0]:,} rows × {prices.shape[1]} columns")
    if 'item_id' in prices.columns:
        info("Unique items", f"{prices['item_id'].nunique()}")
    info("Price range", f"${prices['sell_price'].min():.2f} – ${prices['sell_price'].max():.2f}")
    info("Mean price", f"${prices['sell_price'].mean():.2f}")
    info("Median price", f"${prices['sell_price'].median():.2f}")

    # Save
    out_path = CLEANED_PATH / "sell_prices_cleaned.csv"
    prices.to_csv(out_path, index=False)
    print(f"\n  {c('💾 Saved:', TermColors.GREEN)} {out_path.name} ({len(prices):,} rows)")

    if log_file:
        log_file.write(f"Prices: {original_shape} → {prices.shape}, saved to {out_path}\n")

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

    # Create output directories
    CLEANED_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    # Open log file
    log_path = LOG_PATH / "data_cleaning.log"
    with open(log_path, 'w') as log_file:
        log_file.write(f"M5 Raw Data Cleaning — {datetime.now().isoformat()}\n")
        log_file.write("=" * 60 + "\n\n")

        # Step 1: Clean Calendar
        cal = clean_calendar(log_file)

        # Step 2: Clean Sales
        sales = clean_sales_train_validation(log_file)

        # Get valid item list for price filtering
        valid_items = sales['item_id'].unique().tolist() if 'item_id' in sales.columns else None

        # Step 3: Clean Prices
        prices = clean_sell_prices(valid_items, log_file)

        # ── Final Report ──
        elapsed = time.time() - start_time
        header("CLEANING COMPLETE", "🏁")
        print(f"\n  {c('Files cleaned:', TermColors.BOLD)}")
        print(f"     📅 calendar_cleaned.csv      {cal.shape[0]:>8,} rows")
        print(f"     🛒 sales_cleaned.csv          {sales.shape[0]:>8,} rows")
        print(f"     💰 prices_cleaned.csv         {prices.shape[0]:>8,} rows")
        print(f"\n  {c('Output directory:', TermColors.DIM)}  {CLEANED_PATH}")
        print(f"  {c('Log file:', TermColors.DIM)}          {log_path}")
        print(f"  {c('Time elapsed:', TermColors.DIM)}      {elapsed:.1f}s")
        print()
        print(c("═" * w, TermColors.CYAN))

        # Write final summary to log
        log_file.write(f"\nCleaning complete in {elapsed:.1f}s\n")
        log_file.write(f"Calendar:  {cal.shape[0]:,} rows\n")
        log_file.write(f"Sales:     {sales.shape[0]:,} rows\n")
        log_file.write(f"Prices:    {prices.shape[0]:,} rows\n")

    return cal, sales, prices


if __name__ == "__main__":
    main()
