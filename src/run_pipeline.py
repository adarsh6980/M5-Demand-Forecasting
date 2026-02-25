"""
M5 Forecasting — Console Pipeline Runner
Runs the full demand forecasting pipeline and displays drift detection
results and business rules application in a formatted terminal output.

Usage:
    python src/run_pipeline.py
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from clean_m5_data import main as clean_data
from data_ingestion import load_pos_data, load_stock_data, get_sku_list
from feature_engineering import prepare_features, get_feature_columns
from drift_detection import DriftMonitor
from business_rules import BusinessRuleEngine
from forecasting import ForecastingEngine

import logging

# Suppress ALL logging output during pipeline — we use our own formatted output
# This prevents drift_detection WARNING messages from interleaving with tables
logging.disable(logging.CRITICAL)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
CONFIG_PATH = BASE_PATH / "config"
MODELS_PATH = BASE_PATH / "models"


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


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════

def step_clean_data():
    """Run data cleaning if needed, otherwise load existing."""
    pos_path = DATA_PATH / "pos_data.csv"
    
    if not pos_path.exists():
        header("STEP 1 — DATA CLEANING", "🧹")
        print(f"\n  Cleaning M5 raw data → {pos_path}")
        
        # Temporarily re-enable logging for cleaning step
        logging.disable(logging.NOTSET)
        pos_df, rules = clean_data()
        logging.disable(logging.CRITICAL)
        return pos_df
    else:
        header("STEP 1 — DATA LOADING", "📂")
        pos_df = load_pos_data(str(pos_path))
        
        info("Records", f"{len(pos_df):,}")
        info("SKUs", f"{pos_df['sku'].nunique()}")
        info("Store", pos_df['store_id'].iloc[0] if 'store_id' in pos_df.columns else "N/A")
        info("Date range", f"{pos_df['date'].min().date()} → {pos_df['date'].max().date()}")
        info("Avg daily sales", f"{pos_df['units_sold'].mean():.1f} units")
        info("Price range", f"${pos_df['price'].min():.2f} – ${pos_df['price'].max():.2f}")
        
        return pos_df


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def step_features(pos_df):
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
    
    return features_df, available


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════

def step_train(features_df, feature_cols):
    """Train XGBoost models per SKU."""
    header("STEP 3 — MODEL TRAINING", "🤖")
    
    engine = ForecastingEngine(str(MODELS_PATH))
    skus = features_df['sku'].unique()
    total = len(skus)
    
    print(f"\n  Training {c(str(total), TermColors.BOLD)} SKU models...\n")
    
    # Table header
    print(f"  {'SKU':<30} │ {'MAE':>8} │ {'RMSE':>8} │ {'R²':>7} │ {'Samples':>8}")
    print(f"  {'─' * 30}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 7}─┼─{'─' * 8}")
    
    all_predictions = {}
    
    for i, sku in enumerate(skus):
        sku_df = features_df[features_df['sku'] == sku].copy()
        
        if len(sku_df) < 50:
            continue
        
        try:
            # Suppress XGBoost output
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
            
            # Truncate SKU name for display
            sku_display = sku[:28] + ".." if len(sku) > 30 else sku
            print(f"  {sku_display:<30} │ {mae:>8.2f} │ {rmse:>8.2f} │ {r2_str} │ {n_samples:>8,}")
            
        except Exception as e:
            sku_display = sku[:28] + ".." if len(sku) > 30 else sku
            print(f"  {sku_display:<30} │ {c('FAILED', TermColors.RED):>20} │         │         │")
    
    trained = len([m for m in engine.models.values() if m.model is not None])
    print(f"\n  {c('✅', TermColors.GREEN)} Trained {c(str(trained), TermColors.BOLD)}/{total} models successfully")
    
    return engine, all_predictions


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════

def step_drift_detection(all_predictions):
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
    
    return results_summary


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: BUSINESS RULES APPLICATION
# ═══════════════════════════════════════════════════════════════════════

def step_business_rules(pos_df):
    """Apply business rules and display results."""
    header("STEP 5 — BUSINESS RULES APPLICATION", "📋")
    
    rules_path = CONFIG_PATH / "business_rules.yml"
    if not rules_path.exists():
        print(f"  {c('⚠️  No business rules found. Skipping.', TermColors.YELLOW)}")
        return
    
    engine = BusinessRuleEngine(str(rules_path))
    skus = pos_df['sku'].unique()
    
    # Get stock data (default)
    stock_df = pd.DataFrame({
        'sku': skus,
        'current_stock': np.random.randint(10, 100, len(skus))
    })
    np.random.seed(42)
    
    print(f"\n  Applying {c(str(len(engine.rules)), TermColors.BOLD)} rules to {c(str(len(skus)), TermColors.BOLD)} SKUs...\n")
    
    # Generate some forecast values per SKU
    forecasts = {}
    for sku in skus:
        sku_data = pos_df[pos_df['sku'] == sku]
        avg = sku_data['units_sold'].mean()
        std = sku_data['units_sold'].std()
        # Forecast = recent trend (last 14 days avg * 7 for weekly order)
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


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    
    print()
    print(c("╔" + "═" * 70 + "╗", TermColors.CYAN))
    print(c("║", TermColors.CYAN) + c("  🏪  M5 FORECASTING ACCURACY — FULL PIPELINE", TermColors.BOLD + TermColors.WHITE).center(83) + c("║", TermColors.CYAN))
    print(c("║", TermColors.CYAN) + c("  Walmart Retail  │  Store CA_1  │  Console Output", TermColors.DIM).center(83) + c("║", TermColors.CYAN))
    print(c("╚" + "═" * 70 + "╝", TermColors.CYAN))
    
    # Step 1: Load / Clean data
    progress(1, 5, "Data Loading & Cleaning")
    pos_df = step_clean_data()
    
    # Step 2: Feature Engineering
    progress(2, 5, "Feature Engineering")
    features_df, feature_cols = step_features(pos_df)
    
    # Step 3: Train Models
    progress(3, 5, "Model Training")
    engine, all_predictions = step_train(features_df, feature_cols)
    
    # Step 4: Drift Detection
    progress(4, 5, "Drift Detection")
    drift_results = step_drift_detection(all_predictions)
    
    # Step 5: Business Rules
    progress(5, 5, "Business Rules")
    step_business_rules(pos_df)
    
    # Final summary
    elapsed = time.time() - start_time
    
    header("PIPELINE COMPLETE", "🏁")
    print(f"\n  Total time: {c(f'{elapsed:.1f}s', TermColors.BOLD)}")
    print(f"  Data: {c(f'{len(pos_df):,}', TermColors.CYAN)} records across {c(str(pos_df['sku'].nunique()), TermColors.CYAN)} SKUs")
    if 'store_id' in pos_df.columns:
        print(f"  Store: {c(pos_df['store_id'].iloc[0], TermColors.CYAN)}")
    print(f"  Models trained: {c(str(len(all_predictions)), TermColors.GREEN)}")
    print(f"  Drift events: {c(str(sum(1 for d in drift_results if d['drift'])), TermColors.YELLOW)}")
    print()
    print(c("═" * 72, TermColors.CYAN))
    print()


if __name__ == "__main__":
    main()
