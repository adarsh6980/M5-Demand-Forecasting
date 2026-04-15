"""
M5 Demand Forecasting — Project Image Capture Script
=====================================================
Runs the full pipeline and captures:
  • Terminal outputs  →  terminal_output_1.png … terminal_output_7.png
  • Front-end charts  →  front_end_image_1.png … front_end_image_6.png

ALL images use a **light theme** and are saved to  <project>/project images/

Usage:
    python capture_project_images.py
"""

import sys
import os
import time
import io
import contextlib
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Pillow for terminal‑image rendering ──────────────────────────────
from PIL import Image, ImageDraw, ImageFont

# ── Matplotlib in Agg mode, light theme ──────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("default")                    # explicit light theme
matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor": "white",
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── Project paths ────────────────────────────────────────────────────
BASE_PATH = Path(__file__).parent
sys.path.insert(0, str(BASE_PATH / "src_main"))

OUTPUT_DIR = BASE_PATH / "project images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_PATH = BASE_PATH / "data_cleaned"
CONFIG_PATH  = BASE_PATH / "config1"
MODELS_PATH  = BASE_PATH / "models1"

# ── Import pipeline modules ──────────────────────────────────────────
import logging
logging.disable(logging.CRITICAL)           # silence lib noise

from clean_raw_data import main as clean_data
from data_ingestion import (
    load_calendar, load_sales, load_prices,
    melt_and_merge, load_stock_data, get_sku_list,
)
from feature_engineering import prepare_features, get_feature_columns
from drift_detection import DriftMonitor
from business_rules import BusinessRuleEngine
from forecasting import ForecastingEngine, ForecastModel
import yaml
import concurrent.futures
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
#  1.  LIGHT‑THEMED TERMINAL IMAGE RENDERER
# ═══════════════════════════════════════════════════════════════════════

def _font(size=26):
    for p in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/Library/Fonts/Courier New.ttf",
        "/System/Library/Fonts/Monaco.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _bold_font(size=26):
    for p in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Bold.otf",
        "/Library/Fonts/Courier New Bold.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return _font(size)


# Light‑theme colour map  (dark text on white background)
_COLORS = {
    "WHITE":  "#1e1e1e",
    "GREEN":  "#16a34a",
    "YELLOW": "#b45309",
    "RED":    "#dc2626",
    "CYAN":   "#0891b2",
    "BLUE":   "#2563eb",
    "DIM":    "#9ca3af",
    "BOLD":   "#111827",
    "HEADER": "#7c3aed",
}
_BG       = "#ffffff"
_TITLE_BG = "#f3f4f6"
_TITLE_FG = "#4b5563"


def render_terminal_image(lines, filename, title="Terminal", width=1920):
    """Render a list of (text, colour_key) tuples as a light macOS‑terminal PNG."""
    font      = _font(26)
    bold_font = _bold_font(26)
    title_font = _font(24)

    line_h    = 38
    pad       = 40
    title_h   = 60
    img_h     = title_h + pad * 2 + len(lines) * line_h + 12

    img  = Image.new("RGB", (width, img_h), _BG)
    draw = ImageDraw.Draw(img)

    # title bar
    draw.rectangle([0, 0, width, title_h], fill=_TITLE_BG)
    draw.line([(0, title_h), (width, title_h)], fill="#d1d5db", width=1)
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        cx = 36 + i * 40
        draw.ellipse([cx - 12, 18, cx + 12, 42], fill=c)
    try:
        tw = draw.textlength(title, font=title_font)
    except AttributeError:
        tw = len(title) * 7
    draw.text(((width - tw) / 2, 16), title, fill=_TITLE_FG, font=title_font)

    # body
    y = title_h + pad
    for item in lines:
        if isinstance(item, str):
            text, colour = item, _COLORS["WHITE"]
        else:
            text, cname = item
            colour = _COLORS.get(cname, _COLORS["WHITE"])
        f = bold_font if (isinstance(item, tuple) and item[1] == "BOLD") else font
        draw.text((pad, y), text, fill=colour, font=f)
        y += line_h

    out = OUTPUT_DIR / filename
    img.save(str(out), "PNG")
    print(f"  ✅  Saved {out.name}")


# ═══════════════════════════════════════════════════════════════════════
#  2.  PIPELINE EXECUTION  (collects data for rendering)
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Execute pipeline and return all data needed for image captures."""
    t0 = time.time()

    # ── Load / clean data ─────────────────────────────────────────
    cal_path    = CLEANED_PATH / "calendar_cleaned.csv"
    sales_path  = CLEANED_PATH / "sales_train_validation_cleaned.csv"
    prices_path = CLEANED_PATH / "sell_prices_cleaned.csv"

    if not (cal_path.exists() and sales_path.exists() and prices_path.exists()):
        logging.disable(logging.NOTSET)
        cal, sales, prices = clean_data()
        logging.disable(logging.CRITICAL)
    else:
        cal    = load_calendar(str(cal_path))
        sales  = load_sales(str(sales_path))
        prices = load_prices(str(prices_path))

    pos_df = melt_and_merge(sales, cal, prices)

    # ── Feature engineering ───────────────────────────────────────
    features_df  = prepare_features(pos_df, include_lags=True)
    feature_cols = get_feature_columns()
    available    = [f for f in feature_cols if f in features_df.columns]

    # ── Model training ────────────────────────────────────────────
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    engine = ForecastingEngine(str(MODELS_PATH))
    skus   = features_df["sku"].unique()

    all_predictions   = {}
    training_results  = []

    def _train(sku):
        df = features_df[features_df["sku"] == sku].copy()
        if len(df) < 50:
            return None
        try:
            m = ForecastModel(sku)
            m.train(df, available, use_time_series_cv=False)
            p = m.predict(df)
            return {
                "sku": sku, "model": m,
                "preds": {"actual": df["units_sold"].values, "predicted": p},
                "metrics": m.metrics, "samples": len(df),
            }
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futs = {ex.submit(_train, s): s for s in skus}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            if r is None or "error" in (r or {}):
                continue
            engine.models[r["sku"]] = r["model"]
            all_predictions[r["sku"]] = r["preds"]
            training_results.append({
                "sku": r["sku"],
                "mae":  r["metrics"].get("mae", 0),
                "rmse": r["metrics"].get("rmse", 0),
                "r2":   r["metrics"].get("r2", 0),
                "samples": r["samples"],
            })

    # ── Drift detection ───────────────────────────────────────────
    monitor = DriftMonitor()
    drift_results = []
    for sku, data in all_predictions.items():
        actual    = data["actual"]
        predicted = data["predicted"]
        last = None
        for a, p in zip(actual[-100:], predicted[-100:]):
            last = monitor.update(sku, float(a), float(p))
        if last is None:
            continue
        residuals = actual[-100:] - predicted[-100:]
        drift_results.append({
            "sku": sku,
            "drift": last.get("drift_detected", False),
            "severity": last.get("severity", "none"),
            "mean_residual": float(np.mean(residuals)),
            "std_residual":  float(np.std(residuals)),
            "detectors": ", ".join(last.get("detectors_triggered", [])) or "—",
        })

    # ── Business rules ────────────────────────────────────────────
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    rules_path = CONFIG_PATH / "business_rules.yml"
    np.random.seed(42)

    # generate_business_rules inline
    skus_config = []
    grouped = pos_df.groupby("sku")
    for sku in pos_df["sku"].unique():
        sd = grouped.get_group(sku)
        avg_d = sd["units_sold"].mean()
        max_d = sd["units_sold"].max()
        avg_p = sd["price"].mean()
        cat   = sd["cat_id"].iloc[0] if "cat_id" in sd.columns else "UNKNOWN"
        if cat == "FOODS":
            pdays, ss = int(np.random.choice([7, 10, 14])), 1
        elif cat == "HOUSEHOLD":
            pdays, ss = int(np.random.choice([90, 120, 180])), 3
        else:
            pdays, ss = int(np.random.choice([60, 90, 120])), 2
        skus_config.append({
            "sku": sku, "category": cat,
            "max_shelf_capacity": int(max(max_d * 3, avg_d * 7, 50)),
            "unit_cost": round(float(avg_p * 0.55), 2),
            "max_budget_per_order": int(max(500, avg_d * avg_p * 7)),
            "perishability_days": pdays,
            "safety_stock_days": ss,
        })
    with open(rules_path, "w") as f:
        yaml.dump({"skus": skus_config}, f, default_flow_style=False)

    br_engine = BusinessRuleEngine(str(rules_path))
    np.random.seed(42)
    stock_df = pd.DataFrame({
        "sku": pos_df["sku"].unique(),
        "current_stock": np.random.randint(10, 100, pos_df["sku"].nunique()),
    })

    forecasts = {}
    for sku in pos_df["sku"].unique():
        recent = grouped.get_group(sku).tail(14)["units_sold"].mean()
        forecasts[sku] = int(recent * 7)

    br_rows = []
    total_forecast = total_order = capped = 0
    for sku in pos_df["sku"].unique():
        fc  = forecasts.get(sku, 0)
        stk = int(stock_df.loc[stock_df["sku"] == sku, "current_stock"].iloc[0])
        dd  = grouped.get_group(sku)["units_sold"].mean()
        res = br_engine.apply_rules(
            forecast_qty=float(fc), current_stock=float(stk),
            sku=sku, daily_demand=float(dd),
        )
        fq = res["final_qty"]; ex = res["explanation"]
        total_forecast += fc; total_order += fq
        if "Capped" in ex:
            capped += 1
        br_rows.append({"sku": sku, "forecast": fc, "stock": stk,
                         "order": fq, "rule": ex})

    reduction = ((total_forecast - total_order) / total_forecast * 100) if total_forecast else 0
    elapsed = time.time() - t0

    return {
        "pos_df": pos_df, "cal": cal, "sales": sales, "prices": prices,
        "features_df": features_df, "available": available,
        "training_results": training_results,
        "all_predictions": all_predictions,
        "drift_results": drift_results,
        "br_rows": br_rows,
        "total_forecast": total_forecast, "total_order": total_order,
        "capped": capped, "reduction": reduction,
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
#  3.  TERMINAL IMAGE GENERATION  (terminal_output_1 … 7)
# ═══════════════════════════════════════════════════════════════════════

def generate_terminal_images(data):
    """Render 7 terminal‑style light‑themed PNG screenshots."""
    pos   = data["pos_df"]
    cal   = data["cal"]
    sales = data["sales"]
    prices = data["prices"]
    avail = data["available"]
    tr    = data["training_results"]
    dr    = data["drift_results"]
    br    = data["br_rows"]

    print("\n📸  Generating terminal output images (light theme)…")

    # ── terminal_output_1 : Data Loading ──────────────────────────
    lines = [
        ("  [█░░░░] 20%  Data Cleaning & Loading", "GREEN"), ("", "WHITE"),
        ("═" * 72, "CYAN"),
        ("  📂  STEP 1 — DATA LOADING", "BOLD"),
        ("═" * 72, "CYAN"), ("", "WHITE"),
        ("  Source Files", "YELLOW"),
        (f"     Calendar:    {len(cal):>8,} rows", "WHITE"),
        (f"     Sales:       {len(sales):>8,} items", "WHITE"),
        (f"     Prices:      {len(prices):>8,} records", "WHITE"),
        ("", "WHITE"),
        ("  Merged Dataset", "YELLOW"),
        (f"     Records:     {len(pos):>8,}", "WHITE"),
        (f"     SKUs:        {pos['sku'].nunique():>8}", "WHITE"),
        (f"     Stores:      {pos['store_id'].nunique() if 'store_id' in pos.columns else 'N/A':>8}", "WHITE"),
        (f"     Date range:  {pos['date'].min().date()} → {pos['date'].max().date()}", "WHITE"),
        (f"     Avg daily:   {pos['units_sold'].mean():.1f} units", "WHITE"),
        (f"     Price range: ${pos['price'].min():.2f} – ${pos['price'].max():.2f}", "WHITE"),
    ]
    render_terminal_image(lines, "terminal_output_1.png", "Step 1 — Data Loading")

    # ── terminal_output_2 : Feature Engineering ───────────────────
    dow = pos.groupby(pos["date"].dt.dayofweek)["units_sold"].mean()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines = [
        ("  [██░░░] 40%  Feature Engineering", "GREEN"), ("", "WHITE"),
        ("═" * 72, "CYAN"),
        ("  ⚙️   STEP 2 — FEATURE ENGINEERING", "BOLD"),
        ("═" * 72, "CYAN"), ("", "WHITE"),
        (f"     Input records:   {len(pos):,}", "WHITE"),
        (f"     Output records:  {len(data['features_df']):,}", "WHITE"),
        (f"     Features used:   {len(avail)}", "WHITE"),
        ("", "WHITE"),
        ("  Feature List", "YELLOW"),
    ]
    for i in range(0, len(avail), 4):
        chunk = avail[i:i+4]
        lines.append(("     " + "  │  ".join(f"{f:<18}" for f in chunk), "WHITE"))
    lines += [("", "WHITE"), ("  Day of Week Sales Pattern", "YELLOW")]
    for d_name, val in zip(days, dow.values):
        lines.append((f"     {d_name}: {val:.1f} avg units", "WHITE"))
    render_terminal_image(lines, "terminal_output_2.png", "Step 2 — Feature Engineering")

    # ── terminal_output_3 : Model Training ────────────────────────
    MAX_DISPLAY = 30  # Cap rows to keep image readable
    # Sort by MAE descending for most interesting SKUs first
    tr_sorted = sorted(tr, key=lambda r: r["mae"], reverse=True)
    tr_display = tr_sorted[:MAX_DISPLAY]
    tr_remaining = len(tr) - len(tr_display)

    lines = [
        ("  [███░░] 60%  Model Training", "GREEN"), ("", "WHITE"),
        ("═" * 72, "CYAN"),
        ("  🤖  STEP 3 — MODEL TRAINING", "BOLD"),
        ("═" * 72, "CYAN"), ("", "WHITE"),
        (f"  Training {len(pos['sku'].unique())} SKU models…", "WHITE"),
        ("", "WHITE"),
        ("  SKU                          │      MAE │     RMSE │      R² │  Samples", "DIM"),
        ("  " + "─" * 30 + "─┼─" + "─" * 8 + "─┼─" + "─" * 8 + "─┼─" + "─" * 7 + "─┼─" + "─" * 8, "DIM"),
    ]
    for r in tr_display:
        sku_d = r["sku"][:28] + ".." if len(r["sku"]) > 30 else r["sku"]
        r2_c = "GREEN" if r["r2"] > 0.7 else ("YELLOW" if r["r2"] > 0.3 else "RED")
        lines.append((f"  {sku_d:<30} │ {r['mae']:>8.2f} │ {r['rmse']:>8.2f} │ {r['r2']:>7.3f} │ {r['samples']:>8,}", r2_c if r["r2"] <= 0.3 else "WHITE"))
    if tr_remaining > 0:
        lines.append((f"  … and {tr_remaining} more SKUs (showing top {MAX_DISPLAY} by MAE)", "DIM"))
    trained = len(tr)
    total_skus = len(pos["sku"].unique())
    lines += [
        ("", "WHITE"),
        (f"  ✅ Trained {trained}/{total_skus} models successfully", "GREEN"),
        ("", "WHITE"),
        ("  OVERALL MODEL TRAINING METRICS", "BOLD"),
        (f"     Average MAE:    {np.mean([r['mae'] for r in tr]):.2f}", "WHITE"),
        (f"     Average RMSE:   {np.mean([r['rmse'] for r in tr]):.2f}", "WHITE"),
        (f"     Average R²:     {np.mean([r['r2'] for r in tr]):.3f}", "WHITE"),
        (f"     Good Models:    {sum(1 for r in tr if r['r2']>0.7)}/{trained}", "WHITE"),
    ]
    render_terminal_image(lines, "terminal_output_3.png", "Step 3 — Model Training", width=2040)

    # ── terminal_output_4 : Drift Detection ───────────────────────
    # Show drift/warning SKUs first, then stable (capped at 30 total)
    dr_drift   = [r for r in dr if r["drift"]]
    dr_warning = [r for r in dr if not r["drift"] and r["severity"] == "low"]
    dr_stable  = [r for r in dr if not r["drift"] and r["severity"] != "low"]
    dr_ordered = dr_drift + dr_warning + dr_stable
    dr_display = dr_ordered[:MAX_DISPLAY]
    dr_remaining = len(dr) - len(dr_display)

    lines = [
        ("  [████░] 80%  Drift Detection", "GREEN"), ("", "WHITE"),
        ("═" * 84, "CYAN"),
        ("  📊  STEP 4 — DRIFT DETECTION RESULTS", "BOLD"),
        ("═" * 84, "CYAN"), ("", "WHITE"),
        (f"  Running ADWIN + DDM drift detectors on {len(dr)} SKUs…", "WHITE"),
        ("", "WHITE"),
        ("  SKU                          │ Drift? │  Severity  │  Mean Res │  Std Res │ Detectors", "DIM"),
        ("  " + "─" * 30 + "─┼─" + "─" * 8 + "─┼─" + "─" * 10 + "─┼─" + "─" * 9 + "─┼─" + "─" * 8 + "─┼─" + "─" * 16, "DIM"),
    ]
    drift_cnt = len(dr_drift)
    warn_cnt  = len(dr_warning)
    for r in dr_display:
        sku_d = r["sku"][:28] + ".." if len(r["sku"]) > 30 else r["sku"]
        if r["drift"]:
            icon, sev_str, clr = "⚠️ ", f"{'HIGH' if r['severity']=='high' else 'MEDIUM':^10}", "RED" if r["severity"]=="high" else "YELLOW"
        elif r["severity"] == "low":
            icon, sev_str, clr = "⚡ ", f"{'LOW':^10}", "YELLOW"
        else:
            icon, sev_str, clr = "✅ ", f"{'NONE':^10}", "GREEN"
        lines.append((f"  {sku_d:<30} │  {icon}  │  {sev_str} │ {r['mean_residual']:>+9.2f} │ {r['std_residual']:>8.2f} │ {r['detectors']:<16}", clr))
    if dr_remaining > 0:
        lines.append((f"  … and {dr_remaining} more SKUs (showing top {MAX_DISPLAY})", "DIM"))

    stable = len(dr_stable)
    lines += [
        ("", "WHITE"), ("  " + "─" * 68, "DIM"), ("", "WHITE"),
        ("  SUMMARY", "BOLD"),
        (f"     ● Drift detected:  {drift_cnt:>3}  SKUs", "RED"),
        (f"     ● Warning:         {warn_cnt:>3}  SKUs", "YELLOW"),
        (f"     ● Stable:          {stable:>3}  SKUs", "GREEN"),
        (f"     {'─' * 30}", "DIM"),
        (f"     Total monitored:   {len(dr):>3}  SKUs", "WHITE"),
    ]
    render_terminal_image(lines, "terminal_output_4.png", "Step 4 — Drift Detection", width=2040)

    # ── terminal_output_5 : Business Rules ────────────────────────
    # Show capped SKUs first (most interesting), then normal, capped at 30
    br_capped  = [r for r in br if "Capped" in r["rule"]]
    br_zero    = [r for r in br if r["order"] == 0 and "Capped" not in r["rule"]]
    br_normal  = [r for r in br if r["order"] > 0 and "Capped" not in r["rule"]]
    br_ordered = br_capped + br_normal + br_zero
    br_display = br_ordered[:MAX_DISPLAY]
    br_remaining = len(br) - len(br_display)

    lines = [
        ("  [█████] 100%  Business Rules", "GREEN"), ("", "WHITE"),
        ("═" * 84, "CYAN"),
        ("  📋  STEP 5 — BUSINESS RULES APPLICATION", "BOLD"),
        ("═" * 84, "CYAN"), ("", "WHITE"),
        (f"  Applying rules to {len(br)} SKUs…", "WHITE"),
        ("", "WHITE"),
        ("  SKU                          │  Forecast │  Stock │  Order │ Rule Applied", "DIM"),
        ("  " + "─" * 30 + "─┼─" + "─" * 9 + "─┼─" + "─" * 6 + "─┼─" + "─" * 6 + "─┼─" + "─" * 28, "DIM"),
    ]
    for r in br_display:
        sku_d = r["sku"][:28] + ".." if len(r["sku"]) > 30 else r["sku"]
        clr = "YELLOW" if "Capped" in r["rule"] else ("DIM" if r["order"] == 0 else "GREEN")
        rule_text = r["rule"][:28]
        lines.append((f"  {sku_d:<30} │ {r['forecast']:>9,} │ {r['stock']:>6} │ {r['order']:>6} │ {rule_text}", clr))
    if br_remaining > 0:
        lines.append((f"  … and {br_remaining} more SKUs (showing top {MAX_DISPLAY})", "DIM"))
    lines += [
        ("", "WHITE"), ("  " + "─" * 68, "DIM"), ("", "WHITE"),
        ("  SUMMARY", "BOLD"),
        (f"     Total forecasted demand:   {data['total_forecast']:>10,} units", "CYAN"),
        (f"     Total order quantity:       {data['total_order']:>10,} units", "GREEN"),
        (f"     Reduction from rules:       {data['reduction']:>9.1f}%", "YELLOW"),
        (f"     SKUs capped by rules:       {data['capped']:>10} / {len(br)}", "YELLOW"),
    ]
    render_terminal_image(lines, "terminal_output_5.png", "Step 5 — Business Rules", width=2040)

    # ── terminal_output_6 : Pipeline Summary ──────────────────────
    lines = [
        ("", "WHITE"),
        ("═" * 72, "CYAN"),
        ("  🏁  PIPELINE COMPLETE", "BOLD"),
        ("═" * 72, "CYAN"), ("", "WHITE"),
        ("  DATA SOURCES", "BOLD"),
        (f"     📅 Calendar:     {len(cal):>8,} days", "WHITE"),
        (f"     🛒 Sales:        {len(sales):>8,} items", "WHITE"),
        (f"     💰 Prices:       {len(prices):>8,} records", "WHITE"),
        ("", "WHITE"),
        ("  PIPELINE RESULTS", "BOLD"),
        (f"     Total time:       {data['elapsed']:.1f}s", "WHITE"),
        (f"     Merged records:   {len(pos):,}", "CYAN"),
        (f"     SKUs:             {pos['sku'].nunique()}", "CYAN"),
        (f"     Models trained:   {len(tr)}", "GREEN"),
        (f"     Drift events:     {drift_cnt}", "YELLOW"),
        (f"     Order reduction:  {data['reduction']:.1f}%", "YELLOW"),
        ("", "WHITE"),
        ("  OUTPUT FILES", "BOLD"),
        ("     📁 Cleaned data:  data_cleaned/", "WHITE"),
        ("     📁 Images:        project images/", "WHITE"),
        ("     📁 Models:        models1/", "WHITE"),
        ("     📁 Config:        config1/", "WHITE"),
        ("", "WHITE"),
        ("═" * 72, "CYAN"),
    ]
    render_terminal_image(lines, "terminal_output_6.png", "Pipeline Summary")

    # ── terminal_output_7 : Overall Metrics Dashboard ─────────────
    if tr:
        avg_mae  = np.mean([r["mae"] for r in tr])
        avg_rmse = np.mean([r["rmse"] for r in tr])
        avg_r2   = np.mean([r["r2"] for r in tr])
        good_r2  = sum(1 for r in tr if r["r2"] > 0.7)
    else:
        avg_mae = avg_rmse = avg_r2 = good_r2 = 0

    dr_df = pd.DataFrame(dr) if dr else pd.DataFrame()
    avg_mres = dr_df["mean_residual"].mean() if len(dr_df) else 0
    avg_sres = dr_df["std_residual"].mean()  if len(dr_df) else 0

    lines = [
        ("", "WHITE"),
        ("═" * 72, "CYAN"),
        ("  📊  OVERALL METRICS DASHBOARD", "BOLD"),
        ("═" * 72, "CYAN"), ("", "WHITE"),
        ("  MODEL TRAINING", "YELLOW"),
        (f"     Average MAE:           {avg_mae:.2f}", "WHITE"),
        (f"     Average RMSE:          {avg_rmse:.2f}", "WHITE"),
        (f"     Average R²:            {avg_r2:.3f}", "WHITE"),
        (f"     Good Models (R²>0.7):  {good_r2}/{len(tr)}", "WHITE"),
        ("", "WHITE"),
        ("  DRIFT DETECTION", "YELLOW"),
        (f"     Average Mean Residual: {avg_mres:+.2f}", "WHITE"),
        (f"     Average Std Residual:  {avg_sres:.2f}", "WHITE"),
        (f"     Drift Events:          {drift_cnt}", "WHITE"),
        ("", "WHITE"),
        ("  BUSINESS RULES", "YELLOW"),
        (f"     Forecasted Demand:     {data['total_forecast']:,} units", "WHITE"),
        (f"     Final Order Qty:       {data['total_order']:,} units", "WHITE"),
        (f"     Reduction:             {data['reduction']:.1f}%", "WHITE"),
        (f"     Capped SKUs:           {data['capped']}", "WHITE"),
        ("", "WHITE"),
        ("═" * 72, "CYAN"),
    ]
    render_terminal_image(lines, "terminal_output_7.png", "Overall Metrics Dashboard")


# ═══════════════════════════════════════════════════════════════════════
#  4.  FRONT‑END CHART GENERATION (front_end_image_1 … 6)
# ═══════════════════════════════════════════════════════════════════════

def generate_frontend_images(data):
    """Generate 6 light‑themed analytical chart PNGs."""
    pos   = data["pos_df"]
    tr    = data["training_results"]
    dr    = data["drift_results"]
    preds = data["all_predictions"]

    print("\n🎨  Generating front‑end chart images (light theme)…")

    # ── front_end_image_1 : Data Loading Summary ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Step 1 — Data Loading Summary", fontsize=15, fontweight="bold")

    daily = pos.groupby("date")["units_sold"].sum()
    axes[0].plot(daily.index, daily.values, color="#2563eb", lw=0.7)
    axes[0].set_title("Total Daily Sales"); axes[0].set_xlabel("Date"); axes[0].set_ylabel("Units")
    axes[0].tick_params(axis="x", rotation=45)

    sku_sales = pos.groupby("sku")["units_sold"].sum().sort_values().tail(30)
    axes[1].barh(range(len(sku_sales)), sku_sales.values, color="#16a34a")
    axes[1].set_yticks(range(len(sku_sales))); axes[1].set_yticklabels(sku_sales.index, fontsize=7)
    axes[1].set_title("Top 30 SKUs by Sales"); axes[1].set_xlabel("Total Units")

    axes[2].hist(pos["price"], bins=30, color="#f59e0b", edgecolor="white")
    axes[2].set_title("Price Distribution"); axes[2].set_xlabel("Price ($)"); axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "front_end_image_1.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); print("  ✅  Saved front_end_image_1.png")

    # ── front_end_image_2 : Feature Engineering ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Step 2 — Feature Engineering", fontsize=15, fontweight="bold")

    sample_sku = pos["sku"].unique()[0]
    sd = pos[pos["sku"] == sample_sku].tail(90)
    axes[0].plot(sd["date"], sd["units_sold"], color="#2563eb", lw=1.5, label="Actual")
    axes[0].set_title(f"Sales Pattern: {sample_sku}"); axes[0].legend()
    axes[0].set_xlabel("Date"); axes[0].set_ylabel("Units"); axes[0].tick_params(axis="x", rotation=45)

    dow = pos.groupby(pos["date"].dt.dayofweek)["units_sold"].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(day_names, dow.values, color="#7c3aed")
    axes[1].set_title("Average Sales by Day of Week"); axes[1].set_ylabel("Avg Units")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "front_end_image_2.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); print("  ✅  Saved front_end_image_2.png")

    # ── front_end_image_3 : Model Training Results ────────────────
    if tr:
        tr_df = pd.DataFrame(tr).nlargest(30, "mae")
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle("Step 3 — Model Training Results", fontsize=15, fontweight="bold")

        axes[0].barh(range(len(tr_df)), tr_df["mae"].values, color="#e11d48")
        axes[0].set_yticks(range(len(tr_df))); axes[0].set_yticklabels(tr_df["sku"].values, fontsize=7)
        axes[0].set_title("MAE by SKU"); axes[0].set_xlabel("MAE")

        colors = ["#16a34a" if r > 0.7 else "#f59e0b" if r > 0.3 else "#dc2626" for r in tr_df["r2"]]
        axes[1].bar(range(len(tr_df)), tr_df["r2"].values, color=colors)
        axes[1].set_title("R² Score by SKU"); axes[1].set_ylabel("R²")
        axes[1].axhline(y=0.7, color="green", ls="--", alpha=0.5, label="Good (0.7)")
        axes[1].axhline(y=0.3, color="red",   ls="--", alpha=0.5, label="Poor (0.3)")
        axes[1].legend(fontsize=8)

        axes[2].scatter(tr_df["mae"], tr_df["rmse"], c="#2563eb", s=50, alpha=0.7)
        axes[2].set_xlabel("MAE"); axes[2].set_ylabel("RMSE"); axes[2].set_title("RMSE vs MAE")
        axes[2].plot([0, tr_df["mae"].max()], [0, tr_df["mae"].max()], "r--", alpha=0.3, label="MAE=RMSE")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "front_end_image_3.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(); print("  ✅  Saved front_end_image_3.png")

    # ── front_end_image_4 : Drift Detection ───────────────────────
    if dr:
        dr_df = pd.DataFrame(dr)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Step 4 — Drift Detection Results", fontsize=15, fontweight="bold")

        cts = {"Stable": int((~dr_df["drift"]).sum()), "Drift": int(dr_df["drift"].sum())}
        axes[0].pie(cts.values(), labels=cts.keys(), colors=["#16a34a", "#dc2626"],
                     autopct="%1.0f%%", startangle=90)
        axes[0].set_title("Drift Status Distribution")

        n = min(len(dr_df), 30)
        clrs = ["#dc2626" if d else "#16a34a" for d in dr_df["drift"].values[:n]]
        axes[1].barh(range(n), dr_df["mean_residual"].values[:n], color=clrs)
        axes[1].set_yticks(range(n)); axes[1].set_yticklabels(dr_df["sku"].values[:n], fontsize=7)
        axes[1].set_title("Mean Residual by SKU"); axes[1].set_xlabel("Mean Residual")
        axes[1].axvline(x=0, color="black", lw=0.5)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "front_end_image_4.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(); print("  ✅  Saved front_end_image_4.png")

    # ── front_end_image_5 : Forecast vs Actuals (sample SKU) ──────
    if preds:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle("Forecast vs Actuals — Sample SKU", fontsize=15, fontweight="bold")

        sample = list(preds.keys())[0]
        act  = preds[sample]["actual"][-90:]
        pred = preds[sample]["predicted"][-90:]
        ax.plot(range(len(act)),  act,  label="Actual",   color="#2563eb", lw=1.5)
        ax.plot(range(len(pred)), pred, label="Forecast", color="#f59e0b", lw=1.5, ls="--")
        ax.set_title(f"90‑Day Forecast: {sample}"); ax.set_xlabel("Day"); ax.set_ylabel("Units")
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "front_end_image_5.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(); print("  ✅  Saved front_end_image_5.png")

    # ── front_end_image_6 : Pipeline Summary Bar Chart ────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Pipeline Summary — Key Metrics", fontsize=15, fontweight="bold")

    labels = ["Records", "SKUs", "Models Trained", "Drift Events"]
    vals   = [len(pos), pos["sku"].nunique(), len(tr),
              sum(1 for d in dr if d["drift"])]
    bars = ax.bar(labels, vals, color=["#2563eb", "#16a34a", "#f59e0b", "#dc2626"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                f"{v:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Count"); ax.set_title("Key Pipeline Metrics")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "front_end_image_6.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); print("  ✅  Saved front_end_image_6.png")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  M5 DEMAND FORECASTING — PROJECT IMAGE CAPTURE             ║")
    print("║  Light Theme  •  Terminal + Front‑End  •  project images/  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print("\n⏳  Running pipeline…")
    data = run_pipeline()
    print(f"✅  Pipeline finished in {data['elapsed']:.1f}s\n")

    generate_terminal_images(data)
    generate_frontend_images(data)

    # ── Final manifest ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  📁  ALL IMAGES IN: project images/")
    print("═" * 60)
    files = sorted(OUTPUT_DIR.glob("*.png"))
    for f in files:
        kb = f.stat().st_size / 1024
        print(f"    {f.name:<30}  {kb:>7.1f} KB")
    print(f"\n  Total: {len(files)} images")
    print("═" * 60)
    print("🎉  Done!\n")
