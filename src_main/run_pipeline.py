"""
M5 Forecasting — Raw CSV Pipeline Runner
Runs the full demand forecasting pipeline using the 3 raw M5 data files
(calendar, sales_train_validation, sell_prices) through cleaning,
feature engineering, model training, drift detection, and business rules.

Results are displayed in formatted terminal output.
Logs go to logs1/, images to Images1/.

Usage:
    python src_main/run_pipeline.py
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src_main to path
sys.path.insert(0, str(Path(__file__).parent))

from clean_raw_data import main as clean_data
from data_ingestion import (
    load_calendar, load_sales, load_prices,
    melt_and_merge, load_stock_data, get_sku_list
)
from feature_engineering import prepare_features, get_feature_columns
from drift_detection import DriftMonitor
from business_rules import BusinessRuleEngine
from forecasting import ForecastingEngine

import yaml

# Suppress ALL logging output during pipeline — we use our own formatted output
logging.disable(logging.CRITICAL)

BASE_PATH = Path(__file__).parent.parent
CLEANED_PATH = BASE_PATH / "data_cleaned"
CONFIG_PATH = BASE_PATH / "config1"
MODELS_PATH = BASE_PATH / "models1"
IMAGES_PATH = BASE_PATH / "Images1"
LOGS_PATH = BASE_PATH / "logs1"

TARGET_STORE = "CA_1"
TOP_N_ITEMS = 30


# ═══════════════════════════════════════════════════════════════════════
# TERMINAL FORMATTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

class TermColors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def c(text, color):
    """Wrap text in color codes."""
    return f"{color}{text}{TermColors.RESET}"


def header(title, emoji="", width=72):
    """Print a formatted section header."""
    print()
    print(c("═" * width, TermColors.CYAN))
    if emoji:
        print(c(f"  {emoji}  {title}", TermColors.BOLD + TermColors.WHITE))
    else:
        print(c(f"  {title}", TermColors.BOLD + TermColors.WHITE))
    print(c("═" * width, TermColors.CYAN))


def subheader(title, emoji=""):
    """Print a sub-section header."""
    if emoji:
        print(f"\n  {emoji} {c(title, TermColors.BOLD + TermColors.YELLOW)}")
    else:
        print(f"\n  {c(title, TermColors.BOLD + TermColors.YELLOW)}")


def info(label, value, indent=5):
    """Print a label: value pair."""
    spaces = " " * indent
    print(f"{spaces}{c(label + ':', TermColors.DIM)}  {value}")


def divider(char="─", width=72):
    """Print a thin divider."""
    print(c(f"  {char * (width - 4)}", TermColors.DIM))


def progress(step, total, message):
    """Print a progress step."""
    bar = "█" * step + "░" * (total - step)
    pct = int(step / total * 100)
    print(f"\n  [{c(bar, TermColors.GREEN)}] {c(f'{pct}%', TermColors.BOLD)} {message}")


def write_log(log_file, message):
    """Write a message to the log file."""
    if log_file:
        log_file.write(message + "\n")


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: DATA CLEANING & LOADING
# ═══════════════════════════════════════════════════════════════════════

def step_clean_and_load(log_file=None):
    """Clean raw data if needed, then load and merge the 3 cleaned CSVs."""
    cal_path = CLEANED_PATH / "calendar_cleaned.csv"
    sales_path = CLEANED_PATH / "sales_train_validation_cleaned.csv"
    prices_path = CLEANED_PATH / "sell_prices_cleaned.csv"

    # Check if cleaned data exists
    all_exist = cal_path.exists() and sales_path.exists() and prices_path.exists()

    if not all_exist:
        header("STEP 1 — DATA CLEANING & LOADING", "🧹")
        print(f"\n  Cleaned data not found. Running data cleaning pipeline...")
        write_log(log_file, "Running data cleaning pipeline...")

        # Temporarily re-enable logging for cleaning step
        logging.disable(logging.NOTSET)
        cal, sales, prices = clean_data()
        logging.disable(logging.CRITICAL)

        write_log(log_file, f"  Cleaned calendar: {len(cal):,} rows")
        write_log(log_file, f"  Cleaned sales: {len(sales):,} rows")
        write_log(log_file, f"  Cleaned prices: {len(prices):,} rows")
    else:
        header("STEP 1 — LOADING CLEANED DATA", "📂")
        print(f"\n  Loading from {c('data_cleaned/', TermColors.CYAN)}...")

        cal = load_calendar(str(cal_path))
        sales = load_sales(str(sales_path))
        prices = load_prices(str(prices_path))

    # ── Display source file info ──
    subheader("Source Files")
    info("Calendar", f"{len(cal):,} rows  ({cal_path.name})")
    info("Sales", f"{len(sales):,} items  ({sales_path.name})")
    info("Prices", f"{len(prices):,} records  ({prices_path.name})")

    # ── Merge into unified DataFrame ──
    subheader("Merging Data")
    pos_df = melt_and_merge(sales, cal, prices)

    # ── Display merged data info ──
    subheader("Merged Dataset")
    info("Records", f"{len(pos_df):,}")
    info("SKUs", f"{pos_df['sku'].nunique()}")
    info("Store", pos_df['store_id'].iloc[0] if 'store_id' in pos_df.columns else "N/A")
    info("Date range", f"{pos_df['date'].min().date()} → {pos_df['date'].max().date()}")
    info("Avg daily sales", f"{pos_df['units_sold'].mean():.1f} units")
    info("Price range", f"${pos_df['price'].min():.2f} – ${pos_df['price'].max():.2f}")

    write_log(log_file, f"\nStep 1: Loaded {len(pos_df):,} records, {pos_df['sku'].nunique()} SKUs")
    write_log(log_file, f"  Date range: {pos_df['date'].min().date()} → {pos_df['date'].max().date()}")

    return pos_df, cal, sales, prices


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def step_features(pos_df, log_file=None):
    """Prepare features for modeling."""
    header("STEP 2 — FEATURE ENGINEERING", "⚙️")

    features_df = prepare_features(pos_df, include_lags=True)
    feature_cols = get_feature_columns()

    # Only use features that exist
    available = [f for f in feature_cols if f in features_df.columns]

    info("Input records", f"{len(pos_df):,}")
    info("Output records", f"{len(features_df):,}")
    info("Features used", f"{len(available)}")

    subheader("Feature List")
    for i in range(0, len(available), 4):
        chunk = available[i:i+4]
        row = "     " + "  │  ".join(f"{f:<18}" for f in chunk)
        print(row)

    write_log(log_file, f"\nStep 2: Feature engineering → {len(features_df):,} records, {len(available)} features")

    return features_df, available


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════

def step_train(features_df, feature_cols, log_file=None):
    """Train XGBoost models per SKU."""
    header("STEP 3 — MODEL TRAINING", "🤖")

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    engine = ForecastingEngine(str(MODELS_PATH))
    skus = features_df['sku'].unique()
    total = len(skus)

    print(f"\n  Training {c(str(total), TermColors.BOLD)} SKU models...\n")

    # Table header
    print(f"  {'SKU':<30} │ {'MAE':>8} │ {'RMSE':>8} │ {'R²':>7} │ {'Samples':>8}")
    print(f"  {'─' * 30}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 7}─┼─{'─' * 8}")

    all_predictions = {}
    training_results = []

    for i, sku in enumerate(skus):
        sku_df = features_df[features_df['sku'] == sku].copy()

        if len(sku_df) < 50:
            continue

        try:
            import warnings
            warnings.filterwarnings('ignore')

            model = engine.models.get(sku)
            if model is None:
                from forecasting import ForecastModel
                model = ForecastModel(sku)
                engine.models[sku] = model

            model.train(sku_df, feature_cols, use_time_series_cv=False)

            # Get predictions for drift detection
            preds = model.predict(sku_df)
            if preds is not None:
                all_predictions[sku] = {
                    'actual': sku_df['units_sold'].values,
                    'predicted': preds
                }

            metrics = model.metrics
            mae = metrics.get('mae', 0)
            rmse = metrics.get('rmse', 0)
            r2 = metrics.get('r2', 0)
            n_samples = len(sku_df)

            # Color R² based on quality
            if r2 > 0.7:
                r2_str = c(f"{r2:>7.3f}", TermColors.GREEN)
            elif r2 > 0.3:
                r2_str = c(f"{r2:>7.3f}", TermColors.YELLOW)
            else:
                r2_str = c(f"{r2:>7.3f}", TermColors.RED)

            sku_display = sku[:28] + ".." if len(sku) > 30 else sku
            print(f"  {sku_display:<30} │ {mae:>8.2f} │ {rmse:>8.2f} │ {r2_str} │ {n_samples:>8,}")

            training_results.append({
                'sku': sku, 'mae': mae, 'rmse': rmse, 'r2': r2, 'samples': n_samples
            })

        except Exception as e:
            sku_display = sku[:28] + ".." if len(sku) > 30 else sku
            print(f"  {sku_display:<30} │ {c('FAILED', TermColors.RED):>20} │         │         │")

    trained = len([m for m in engine.models.values() if m.model is not None])
    print(f"\n  {c('✅', TermColors.GREEN)} Trained {c(str(trained), TermColors.BOLD)}/{total} models successfully")

    write_log(log_file, f"\nStep 3: Trained {trained}/{total} models")
    for r in training_results:
        write_log(log_file, f"  {r['sku']}: MAE={r['mae']:.2f}, RMSE={r['rmse']:.2f}, R²={r['r2']:.3f}")

    return engine, all_predictions, training_results


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════

def step_drift_detection(all_predictions, log_file=None):
    """Run drift detection and display results."""
    header("STEP 4 — DRIFT DETECTION RESULTS", "📊")

    monitor = DriftMonitor()

    print(f"\n  Running ADWIN + DDM drift detectors on {len(all_predictions)} SKUs...\n")

    # Table header
    print(f"  {'SKU':<30} │ {'Drift?':^8} │ {'Severity':^10} │ {'Mean Res':>9} │ {'Std Res':>8} │ {'Detectors':<16}")
    print(f"  {'─' * 30}─┼─{'─' * 8}─┼─{'─' * 10}─┼─{'─' * 9}─┼─{'─' * 8}─┼─{'─' * 16}")

    drift_count = 0
    warning_count = 0
    results_summary = []

    for sku, data in all_predictions.items():
        actual = data['actual']
        predicted = data['predicted']

        # Feed residuals to drift detector
        detector = monitor.get_detector(sku)
        last_result = None

        for a, p in zip(actual[-100:], predicted[-100:]):
            result = monitor.update(sku, float(a), float(p))
            last_result = result

        if last_result is None:
            continue

        # Calculate statistics
        residuals = actual[-100:] - predicted[-100:]
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)

        drift = last_result.get('drift_detected', False)
        severity = last_result.get('severity', 'none')
        detectors = last_result.get('detectors_triggered', [])

        # Format drift indicator
        if drift:
            drift_count += 1
            drift_str = c("  ⚠️  ", TermColors.RED)
            if severity == 'high':
                sev_str = c(f"{'HIGH':^10}", TermColors.RED + TermColors.BOLD)
            else:
                sev_str = c(f"{'MEDIUM':^10}", TermColors.YELLOW)
        else:
            sev = last_result.get('severity', 'none')
            if sev == 'low':
                warning_count += 1
                drift_str = c("  ⚡  ", TermColors.YELLOW)
                sev_str = c(f"{'LOW':^10}", TermColors.YELLOW)
            else:
                drift_str = c("  ✅  ", TermColors.GREEN)
                sev_str = c(f"{'NONE':^10}", TermColors.GREEN)

        det_str = ", ".join(detectors) if detectors else "—"

        sku_display = sku[:28] + ".." if len(sku) > 30 else sku
        print(f"  {sku_display:<30} │ {drift_str} │ {sev_str} │ {mean_res:>+9.2f} │ {std_res:>8.2f} │ {det_str:<16}")

        results_summary.append({
            'sku': sku, 'drift': drift, 'severity': severity,
            'mean_residual': mean_res, 'std_residual': std_res,
            'detectors': det_str
        })

    # Summary
    total = len(results_summary)
    stable = total - drift_count - warning_count

    print()
    divider()
    print(f"\n  {c('SUMMARY', TermColors.BOLD + TermColors.WHITE)}")
    print(f"     {c('●', TermColors.RED)} Drift detected:   {c(str(drift_count), TermColors.RED + TermColors.BOLD):>3}  SKUs")
    print(f"     {c('●', TermColors.YELLOW)} Warning:           {c(str(warning_count), TermColors.YELLOW):>3}  SKUs")
    print(f"     {c('●', TermColors.GREEN)} Stable:            {c(str(stable), TermColors.GREEN + TermColors.BOLD):>3}  SKUs")
    print(f"     {'─' * 30}")
    print(f"     Total monitored:      {c(str(total), TermColors.BOLD):>3}  SKUs")

    write_log(log_file, f"\nStep 4: Drift Detection")
    write_log(log_file, f"  Drift detected: {drift_count} SKUs")
    write_log(log_file, f"  Warning: {warning_count} SKUs")
    write_log(log_file, f"  Stable: {stable} SKUs")
    for r in results_summary:
        write_log(log_file, f"  {r['sku']}: drift={r['drift']}, severity={r['severity']}, mean_res={r['mean_residual']:.2f}")

    return results_summary


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: BUSINESS RULES APPLICATION
# ═══════════════════════════════════════════════════════════════════════

def generate_business_rules(pos_df):
    """Generate business rules YAML from the merged data."""
    skus_config = []
    for sku in pos_df['sku'].unique():
        sku_data = pos_df[pos_df['sku'] == sku]
        avg_daily = sku_data['units_sold'].mean()
        max_daily = sku_data['units_sold'].max()
        avg_price = sku_data['price'].mean()
        cat = sku_data['cat_id'].iloc[0] if 'cat_id' in sku_data.columns else 'UNKNOWN'

        # Category-specific perishability
        if cat == 'FOODS':
            perishability = np.random.choice([7, 10, 14])
            safety_stock = 1
        elif cat == 'HOUSEHOLD':
            perishability = np.random.choice([90, 120, 180])
            safety_stock = 3
        else:  # HOBBIES or unknown
            perishability = np.random.choice([60, 90, 120])
            safety_stock = 2

        config = {
            'sku': sku,
            'category': cat,
            'max_shelf_capacity': int(max(max_daily * 3, avg_daily * 7, 50)),
            'unit_cost': round(float(avg_price * 0.55), 2),
            'max_budget_per_order': int(max(500, avg_daily * avg_price * 7)),
            'perishability_days': int(perishability),
            'safety_stock_days': safety_stock
        }
        skus_config.append(config)

    return {'skus': skus_config}


def step_business_rules(pos_df, log_file=None):
    """Apply business rules and display results."""
    header("STEP 5 — BUSINESS RULES APPLICATION", "📋")

    # Generate and save business rules
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    rules_path = CONFIG_PATH / "business_rules.yml"

    np.random.seed(42)
    rules = generate_business_rules(pos_df)
    with open(rules_path, 'w') as f:
        yaml.dump(rules, f, default_flow_style=False)

    engine = BusinessRuleEngine(str(rules_path))
    skus = pos_df['sku'].unique()

    # Get stock data (default)
    np.random.seed(42)
    stock_df = pd.DataFrame({
        'sku': skus,
        'current_stock': np.random.randint(10, 100, len(skus))
    })

    print(f"\n  Applying {c(str(len(engine.rules)), TermColors.BOLD)} rules to {c(str(len(skus)), TermColors.BOLD)} SKUs...\n")

    # Generate forecast values per SKU
    forecasts = {}
    for sku in skus:
        sku_data = pos_df[pos_df['sku'] == sku]
        recent = sku_data.tail(14)['units_sold'].mean()
        forecasts[sku] = int(recent * 7)

    # Table header
    print(f"  {'SKU':<30} │ {'Forecast':>9} │ {'Stock':>6} │ {'Order':>6} │ {'Rule Applied':<28}")
    print(f"  {'─' * 30}─┼─{'─' * 9}─┼─{'─' * 6}─┼─{'─' * 6}─┼─{'─' * 28}")

    total_forecast = 0
    total_order = 0
    capped_count = 0

    for sku in skus:
        forecast = forecasts.get(sku, 0)
        stock_row = stock_df[stock_df['sku'] == sku]
        current_stock = int(stock_row['current_stock'].iloc[0]) if len(stock_row) > 0 else 0

        daily_demand = pos_df[pos_df['sku'] == sku]['units_sold'].mean()

        result = engine.apply_rules(
            forecast_qty=float(forecast),
            current_stock=float(current_stock),
            sku=sku,
            daily_demand=float(daily_demand)
        )

        final_qty = result['final_qty']
        explanation = result['explanation']

        total_forecast += forecast
        total_order += final_qty

        # Color based on rule outcome
        if 'Capped' in explanation:
            capped_count += 1
            order_str = c(f"{final_qty:>6}", TermColors.YELLOW)
            rule_str = c(explanation[:28], TermColors.YELLOW)
        elif final_qty == 0:
            order_str = c(f"{final_qty:>6}", TermColors.DIM)
            rule_str = c("Sufficient stock", TermColors.DIM)
        else:
            order_str = c(f"{final_qty:>6}", TermColors.GREEN)
            rule_str = c(explanation[:28], TermColors.GREEN)

        sku_display = sku[:28] + ".." if len(sku) > 30 else sku
        print(f"  {sku_display:<30} │ {forecast:>9,} │ {current_stock:>6} │ {order_str} │ {rule_str}")

    # Summary
    print()
    divider()
    reduction = ((total_forecast - total_order) / total_forecast * 100) if total_forecast > 0 else 0

    print(f"\n  {c('SUMMARY', TermColors.BOLD + TermColors.WHITE)}")
    print(f"     Total forecasted demand:   {c(f'{total_forecast:>10,}', TermColors.CYAN)} units")
    print(f"     Total order quantity:       {c(f'{total_order:>10,}', TermColors.GREEN)} units")
    print(f"     Reduction from rules:       {c(f'{reduction:>9.1f}%', TermColors.YELLOW)}")
    print(f"     SKUs capped by rules:       {c(f'{capped_count:>10}', TermColors.YELLOW)} / {len(skus)}")

    write_log(log_file, f"\nStep 5: Business Rules")
    write_log(log_file, f"  Total forecast: {total_forecast:,}")
    write_log(log_file, f"  Total order: {total_order:,}")
    write_log(log_file, f"  Reduction: {reduction:.1f}%")
    write_log(log_file, f"  SKUs capped: {capped_count}/{len(skus)}")

    return reduction


# ═══════════════════════════════════════════════════════════════════════
# IMAGE GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_images(pos_df, training_results, drift_results, log_file=None):
    """Generate pipeline visualization images to Images1/."""
    header("GENERATING IMAGES", "🖼️")
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Data Loading Summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Step 1 — Data Loading Summary (Raw CSV Pipeline)', fontsize=14, fontweight='bold')

    # Sales distribution
    daily_sales = pos_df.groupby('date')['units_sold'].sum()
    axes[0].plot(daily_sales.index, daily_sales.values, color='#2196F3', linewidth=0.5)
    axes[0].set_title('Total Daily Sales')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Units Sold')
    axes[0].tick_params(axis='x', rotation=45)

    # SKU sales distribution
    sku_sales = pos_df.groupby('sku')['units_sold'].sum().sort_values(ascending=True)
    axes[1].barh(range(len(sku_sales)), sku_sales.values, color='#4CAF50')
    axes[1].set_yticks(range(len(sku_sales)))
    axes[1].set_yticklabels(sku_sales.index, fontsize=6)
    axes[1].set_title('Total Sales by SKU')
    axes[1].set_xlabel('Total Units')

    # Price distribution
    axes[2].hist(pos_df['price'], bins=30, color='#FF9800', edgecolor='white')
    axes[2].set_title('Price Distribution')
    axes[2].set_xlabel('Price ($)')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(IMAGES_PATH / 'step1_data_loading.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {c('✅', TermColors.GREEN)} step1_data_loading.png")

    # 2. Feature Engineering
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Step 2 — Feature Engineering (Raw CSV Pipeline)', fontsize=14, fontweight='bold')

    sample_sku = pos_df['sku'].unique()[0]
    sku_data = pos_df[pos_df['sku'] == sample_sku].tail(90)
    axes[0].plot(sku_data['date'], sku_data['units_sold'], label='Actual', color='#2196F3')
    axes[0].set_title(f'Sales Pattern: {sample_sku}')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Units')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()

    # Day of week pattern
    dow_sales = pos_df.groupby(pos_df['date'].dt.dayofweek)['units_sold'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1].bar(days, dow_sales.values, color='#9C27B0')
    axes[1].set_title('Average Sales by Day of Week')
    axes[1].set_ylabel('Avg Units')

    plt.tight_layout()
    plt.savefig(IMAGES_PATH / 'step2_feature_engineering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {c('✅', TermColors.GREEN)} step2_feature_engineering.png")

    # 3. Model Training
    if training_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Step 3 — Model Training Results (Raw CSV Pipeline)', fontsize=14, fontweight='bold')

        tr_df = pd.DataFrame(training_results)

        # MAE by SKU
        axes[0].barh(range(len(tr_df)), tr_df['mae'].values, color='#E91E63')
        axes[0].set_yticks(range(len(tr_df)))
        axes[0].set_yticklabels(tr_df['sku'].values, fontsize=6)
        axes[0].set_title('MAE by SKU')
        axes[0].set_xlabel('MAE')

        # R² distribution
        colors = ['#4CAF50' if r > 0.7 else '#FFC107' if r > 0.3 else '#F44336' for r in tr_df['r2']]
        axes[1].bar(range(len(tr_df)), tr_df['r2'].values, color=colors)
        axes[1].set_title('R² Score by SKU')
        axes[1].set_ylabel('R²')
        axes[1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
        axes[1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Poor (0.3)')
        axes[1].legend(fontsize=8)

        # RMSE vs MAE scatter
        axes[2].scatter(tr_df['mae'], tr_df['rmse'], c='#2196F3', s=50, alpha=0.7)
        axes[2].set_xlabel('MAE')
        axes[2].set_ylabel('RMSE')
        axes[2].set_title('RMSE vs MAE')
        axes[2].plot([0, tr_df['mae'].max()], [0, tr_df['mae'].max()], 'r--', alpha=0.3, label='MAE=RMSE')
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(IMAGES_PATH / 'step3_model_training.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {c('✅', TermColors.GREEN)} step3_model_training.png")

    # 4. Drift Detection
    if drift_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Step 4 — Drift Detection (Raw CSV Pipeline)', fontsize=14, fontweight='bold')

        dr_df = pd.DataFrame(drift_results)

        # Drift status pie
        status_counts = {
            'Stable': len(dr_df[~dr_df['drift']]),
            'Drift': len(dr_df[dr_df['drift']])
        }
        colors_pie = ['#4CAF50', '#F44336']
        axes[0].pie(status_counts.values(), labels=status_counts.keys(),
                     colors=colors_pie, autopct='%1.0f%%', startangle=90)
        axes[0].set_title('Drift Status Distribution')

        # Mean residual by SKU
        axes[1].barh(range(len(dr_df)), dr_df['mean_residual'].values,
                      color=['#F44336' if d else '#4CAF50' for d in dr_df['drift']])
        axes[1].set_yticks(range(len(dr_df)))
        axes[1].set_yticklabels(dr_df['sku'].values, fontsize=6)
        axes[1].set_title('Mean Residual by SKU')
        axes[1].set_xlabel('Mean Residual')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(IMAGES_PATH / 'step4_drift_detection.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {c('✅', TermColors.GREEN)} step4_drift_detection.png")

    # 5. Pipeline Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Pipeline Summary — Raw CSV Data Pipeline', fontsize=14, fontweight='bold')

    summary_data = {
        'Records': len(pos_df),
        'SKUs': pos_df['sku'].nunique(),
        'Models Trained': len(training_results) if training_results else 0,
        'Drift Events': sum(1 for d in drift_results if d['drift']) if drift_results else 0,
    }

    bars = ax.bar(summary_data.keys(), summary_data.values(),
                   color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])

    for bar, val in zip(bars, summary_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(summary_data.values())*0.02,
                f'{val:,}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Count')
    ax.set_title('Key Pipeline Metrics')

    plt.tight_layout()
    plt.savefig(IMAGES_PATH / 'pipeline_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {c('✅', TermColors.GREEN)} pipeline_summary.png")

    write_log(log_file, f"\nImages saved to {IMAGES_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    # Create output directories
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # Open log file
    log_path = LOGS_PATH / "pipeline_run.log"
    log_file = open(log_path, 'w')
    log_file.write(f"M5 Raw CSV Pipeline Run — {datetime.now().isoformat()}\n")
    log_file.write("=" * 60 + "\n")

    print()
    print(c("╔" + "═" * 70 + "╗", TermColors.CYAN))
    print(c("║", TermColors.CYAN) + c("  🏪  M5 FORECASTING — RAW CSV DATA PIPELINE", TermColors.BOLD + TermColors.WHITE).center(83) + c("║", TermColors.CYAN))
    print(c("║", TermColors.CYAN) + c("  Calendar │ Sales │ Prices  │  Store CA_1  │  Console Output", TermColors.DIM).center(83) + c("║", TermColors.CYAN))
    print(c("╚" + "═" * 70 + "╝", TermColors.CYAN))

    # Step 1: Clean & Load data
    progress(1, 5, "Data Cleaning & Loading")
    pos_df, cal, sales, prices = step_clean_and_load(log_file)

    # Step 2: Feature Engineering
    progress(2, 5, "Feature Engineering")
    features_df, feature_cols = step_features(pos_df, log_file)

    # Step 3: Train Models
    progress(3, 5, "Model Training")
    engine, all_predictions, training_results = step_train(features_df, feature_cols, log_file)

    # Step 4: Drift Detection
    progress(4, 5, "Drift Detection")
    drift_results = step_drift_detection(all_predictions, log_file)

    # Step 5: Business Rules
    progress(5, 5, "Business Rules")
    reduction = step_business_rules(pos_df, log_file)

    # Generate Images
    generate_images(pos_df, training_results, drift_results, log_file)

    # Final summary
    elapsed = time.time() - start_time

    header("PIPELINE COMPLETE", "🏁")
    print(f"\n  {c('DATA SOURCES', TermColors.BOLD + TermColors.WHITE)}")
    print(f"     📅 Calendar:     {len(cal):>8,} days")
    print(f"     🛒 Sales:        {len(sales):>8,} items (wide-format)")
    print(f"     💰 Prices:       {len(prices):>8,} records")

    print(f"\n  {c('PIPELINE RESULTS', TermColors.BOLD + TermColors.WHITE)}")
    print(f"     Total time:       {c(f'{elapsed:.1f}s', TermColors.BOLD)}")
    print(f"     Merged records:   {c(f'{len(pos_df):,}', TermColors.CYAN)}")
    print(f"     SKUs:             {c(str(pos_df['sku'].nunique()), TermColors.CYAN)}")
    if 'store_id' in pos_df.columns:
        print(f"     Store:            {c(pos_df['store_id'].iloc[0], TermColors.CYAN)}")
    print(f"     Models trained:   {c(str(len(all_predictions)), TermColors.GREEN)}")
    print(f"     Drift events:     {c(str(sum(1 for d in drift_results if d['drift'])), TermColors.YELLOW)}")
    print(f"     Order reduction:  {c(f'{reduction:.1f}%', TermColors.YELLOW)}")

    print(f"\n  {c('OUTPUT FILES', TermColors.BOLD + TermColors.WHITE)}")
    print(f"     📁 Cleaned data:  data_cleaned/")
    print(f"     📁 Images:        Images1/")
    print(f"     📁 Logs:          logs1/")
    print(f"     📁 Models:        models1/")
    print(f"     📁 Config:        config1/")
    print()
    print(c("═" * 72, TermColors.CYAN))
    print()

    # Finalize log
    log_file.write(f"\n{'=' * 60}\n")
    log_file.write(f"Pipeline complete in {elapsed:.1f}s\n")
    log_file.write(f"Records: {len(pos_df):,}, SKUs: {pos_df['sku'].nunique()}\n")
    log_file.write(f"Models: {len(all_predictions)}, Drift events: {sum(1 for d in drift_results if d['drift'])}\n")
    log_file.write(f"Order reduction: {reduction:.1f}%\n")
    log_file.close()


if __name__ == "__main__":
    main()
