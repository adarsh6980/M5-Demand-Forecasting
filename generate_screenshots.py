"""
Generate terminal-style PNG screenshots for each pipeline step.
Renders each stage as a dark-themed macOS terminal window image.

Usage:
    python generate_screenshots.py
"""
from PIL import Image, ImageDraw, ImageFont
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.expanduser("~/Downloads")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Font helpers ──────────────────────────────────────────────────────
def get_font(size=14):
    for path in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/Library/Fonts/Courier New.ttf",
        "/System/Library/Fonts/Monaco.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def get_bold_font(size=14):
    for path in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Bold.otf",
        "/Library/Fonts/Courier New Bold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return get_font(size)


# ── Colour palette (Catppuccin-inspired) ──────────────────────────────
COLOR_MAP = {
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

BG       = "#ffffff"
TITLE_BG = "#f3f4f6"
TITLE_FG = "#4b5563"


def render_terminal(lines, filename, title="Terminal", width=1920):
    """Render coloured lines as a high-res macOS-style terminal PNG."""
    font      = get_font(26)
    bold_font = get_bold_font(26)
    title_font = get_font(24)

    line_height    = 38
    padding        = 40
    title_bar_h    = 60

    img_h = title_bar_h + padding * 2 + len(lines) * line_height + 12

    img  = Image.new("RGB", (width, img_h), BG)
    draw = ImageDraw.Draw(img)

    # ── subtle bottom border on title bar ─────────────────────────
    draw.rectangle([0, 0, width, title_bar_h], fill=TITLE_BG)
    draw.line([(0, title_bar_h), (width, title_bar_h)], fill="#d1d5db", width=1)
    # traffic-light dots
    for i, colour in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        cx = 36 + i * 40
        draw.ellipse([cx - 12, 18, cx + 12, 42], fill=colour)
    # centred title
    try:
        tw = draw.textlength(title, font=title_font)
    except AttributeError:
        tw = len(title) * 7
    draw.text(((width - tw) / 2, 16), title, fill=TITLE_FG, font=title_font)

    # ── body ──────────────────────────────────────────────────────
    y = title_bar_h + padding
    for item in lines:
        if isinstance(item, str):
            text, colour = item, COLOR_MAP["WHITE"]
        else:
            text, cname = item
            colour = COLOR_MAP.get(cname, COLOR_MAP["WHITE"])
        f = bold_font if (isinstance(item, tuple) and item[1] == "BOLD") else font
        draw.text((padding, y), text, fill=colour, font=f)
        y += line_height

    out = os.path.join(OUTPUT_DIR, filename)
    img.save(out, "PNG")
    print(f"  ✅  {out}")


# ═════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA LOADING
# ═════════════════════════════════════════════════════════════════════
step1 = [
    ("  [█░░░░] 20% Data Loading & Cleaning", "GREEN"),
    ("", "WHITE"),
    ("════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("  📂  STEP 1 — DATA LOADING", "BOLD"),
    ("════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("", "WHITE"),
    ("     Records:       56,466", "WHITE"),
    ("     SKUs:          30", "WHITE"),
    ("     Store:         CA_1", "WHITE"),
    ("     Date range:    2012-01-29 → 2016-05-22", "WHITE"),
    ("     Avg daily sales:  6.3 units", "WHITE"),
    ("     Price range:   $0.76 – $14.18", "WHITE"),
]
render_terminal(step1, "step1_data_loading.png", "Step 1 — Data Loading", width=1920)


# ═════════════════════════════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════
step2 = [
    ("  [██░░░] 40% Feature Engineering", "GREEN"),
    ("", "WHITE"),
    ("════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("  ⚙️   STEP 2 — FEATURE ENGINEERING", "BOLD"),
    ("════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("", "WHITE"),
    ("     Input records:   56,466", "WHITE"),
    ("     Output records:  56,466", "WHITE"),
    ("     Features used:   14", "WHITE"),
    ("", "WHITE"),
    ("  Feature List", "YELLOW"),
    ("     day_of_week       │  day_of_month      │  month             │  week_of_year", "WHITE"),
    ("     is_weekend        │  is_holiday         │  promo_flag        │  lag_1", "WHITE"),
    ("     lag_7             │  lag_14             │  rolling_mean_7    │  rolling_std_7", "WHITE"),
    ("     rolling_mean_14   │  rolling_std_14", "WHITE"),
]
render_terminal(step2, "step2_feature_engineering.png", "Step 2 — Feature Engineering", width=1920)


# ═════════════════════════════════════════════════════════════════════
#  STEP 3 — MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════
step3 = [
    ("  [███░░] 60% Model Training", "GREEN"),
    ("", "WHITE"),
    ("════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("  🤖  STEP 3 — MODEL TRAINING", "BOLD"),
    ("════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("", "WHITE"),
    ("  Training 30 SKU models...", "WHITE"),
    ("", "WHITE"),
    ("  SKU                          │      MAE │     RMSE │      R² │  Samples", "DIM"),
    ("  ─────────────────────────────┼──────────┼──────────┼─────────┼─────────", "DIM"),
    ("  FOODS_2_019                  │     3.71 │     5.09 │   0.363 │    1,883", "WHITE"),
    ("  FOODS_2_197                  │     7.59 │    11.33 │   0.398 │    1,883", "WHITE"),
    ("  FOODS_2_276                  │     3.81 │     5.44 │   0.321 │    1,883", "WHITE"),
    ("  FOODS_2_347                  │     3.61 │     4.72 │   0.173 │    1,883", "WHITE"),
    ("  FOODS_3_064                  │     9.84 │    13.62 │   0.473 │    1,883", "WHITE"),
    ("  FOODS_3_080                  │     5.31 │     7.62 │   0.370 │    1,883", "WHITE"),
    ("  FOODS_3_090                  │    10.32 │    15.22 │   0.448 │    1,883", "WHITE"),
    ("  FOODS_3_099                  │     4.60 │     6.23 │   0.241 │    1,883", "WHITE"),
    ("  FOODS_3_120                  │    16.79 │    24.50 │   0.419 │    1,883", "WHITE"),
    ("  FOODS_3_202                  │     3.40 │     4.91 │   0.328 │    1,883", "WHITE"),
    ("  FOODS_3_252                  │     6.24 │     8.90 │   0.519 │    1,883", "WHITE"),
    ("  FOODS_3_282                  │     7.87 │    11.13 │   0.370 │    1,883", "WHITE"),
    ("  FOODS_3_295                  │     8.05 │    11.95 │   0.312 │    1,883", "WHITE"),
    ("  FOODS_3_318                  │     3.48 │     5.07 │   0.228 │    1,883", "WHITE"),
    ("  FOODS_3_319                  │     2.08 │     3.24 │   0.243 │    1,883", "WHITE"),
    ("  FOODS_3_491                  │     4.30 │     6.14 │   0.334 │    1,883", "WHITE"),
    ("  FOODS_3_501                  │     8.17 │    12.14 │   0.462 │    1,883", "WHITE"),
    ("  FOODS_3_541                  │     2.85 │     4.34 │   0.305 │    1,883", "WHITE"),
    ("  FOODS_3_555                  │     2.95 │     4.55 │   0.245 │    1,883", "WHITE"),
    ("  FOODS_3_586                  │     5.63 │     7.98 │   0.439 │    1,883", "WHITE"),
    ("  FOODS_3_587                  │     6.48 │     9.65 │   0.328 │    1,883", "WHITE"),
    ("  FOODS_3_607                  │     3.90 │     5.77 │   0.335 │    1,883", "WHITE"),
    ("  FOODS_3_635                  │     0.99 │     1.70 │   0.253 │    1,883", "WHITE"),
    ("  FOODS_3_681                  │     4.74 │     7.02 │   0.382 │    1,883", "WHITE"),
    ("  FOODS_3_694                  │     2.25 │     3.44 │   0.239 │    1,883", "WHITE"),
    ("  FOODS_3_714                  │     4.27 │     6.09 │   0.330 │    1,883", "WHITE"),
    ("  FOODS_3_723                  │     3.87 │     5.95 │   0.245 │    1,883", "WHITE"),
    ("  FOODS_3_741                  │     5.26 │     7.42 │   0.332 │    1,883", "WHITE"),
    ("  FOODS_3_785                  │     7.27 │    10.74 │   0.348 │    1,883", "WHITE"),
    ("  FOODS_3_808                  │     2.19 │     3.46 │   0.280 │    1,883", "WHITE"),
    ("", "WHITE"),
    ("  ✅ Trained 30/30 models successfully", "GREEN"),
]
render_terminal(step3, "step3_model_training.png", "Step 3 — Model Training", width=1920)


# ═════════════════════════════════════════════════════════════════════
#  STEP 4 — DRIFT DETECTION
# ═════════════════════════════════════════════════════════════════════
step4 = [
    ("  [████░] 80% Drift Detection", "GREEN"),
    ("", "WHITE"),
    ("════════════════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("  📊  STEP 4 — DRIFT DETECTION RESULTS", "BOLD"),
    ("════════════════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("", "WHITE"),
    ("  Running ADWIN + DDM drift detectors on 30 SKUs...", "WHITE"),
    ("", "WHITE"),
    ("  SKU                          │ Drift? │  Severity  │  Mean Res │  Std Res │ Detectors", "DIM"),
    ("  ─────────────────────────────┼────────┼────────────┼───────────┼──────────┼──────────────", "DIM"),
    ("  FOODS_2_019                  │   ✅   │    NONE    │     -2.40 │     5.10 │ —", "GREEN"),
    ("  FOODS_2_197                  │   ✅   │    NONE    │     -4.65 │    11.56 │ —", "GREEN"),
    ("  FOODS_2_276                  │   ✅   │    NONE    │     -2.14 │     5.47 │ —", "GREEN"),
    ("  FOODS_2_347                  │   ✅   │    NONE    │     -2.07 │     4.81 │ —", "GREEN"),
    ("  FOODS_3_064                  │   ✅   │    NONE    │     -7.43 │    12.71 │ —", "GREEN"),
    ("  FOODS_3_080                  │   ✅   │    NONE    │     -3.16 │     7.27 │ —", "GREEN"),
    ("  FOODS_3_090                  │   ✅   │    NONE    │    -10.02 │    15.35 │ —", "GREEN"),
    ("  FOODS_3_099                  │   ✅   │    NONE    │     -1.04 │     6.21 │ —", "GREEN"),
    ("  FOODS_3_120                  │   ✅   │    NONE    │     -2.14 │    24.63 │ —", "GREEN"),
    ("  FOODS_3_202                  │   ⚠️   │   MEDIUM   │     +0.06 │     4.91 │ ADWIN", "YELLOW"),
    ("  FOODS_3_252                  │   ✅   │    NONE    │     -0.37 │     8.85 │ —", "GREEN"),
    ("  FOODS_3_282                  │   ✅   │    NONE    │     -2.18 │    11.27 │ —", "GREEN"),
    ("  FOODS_3_295                  │   ✅   │    NONE    │     +1.05 │    11.86 │ —", "GREEN"),
    ("  FOODS_3_318                  │   ✅   │    NONE    │     -2.96 │     5.20 │ —", "GREEN"),
    ("  FOODS_3_319                  │   ✅   │    NONE    │     -0.47 │     3.28 │ —", "GREEN"),
    ("  FOODS_3_491                  │   ✅   │    NONE    │     -3.50 │     6.30 │ —", "GREEN"),
    ("  FOODS_3_501                  │   ✅   │    NONE    │     +1.52 │    13.50 │ —", "GREEN"),
    ("  FOODS_3_541                  │   ✅   │    NONE    │     -3.53 │     4.68 │ —", "GREEN"),
    ("  FOODS_3_555                  │   ✅   │    NONE    │     +0.14 │     4.59 │ —", "GREEN"),
    ("  FOODS_3_586                  │   ✅   │    NONE    │     -0.59 │     7.84 │ —", "GREEN"),
    ("  FOODS_3_587                  │   ✅   │    NONE    │     -2.41 │     9.60 │ —", "GREEN"),
    ("  FOODS_3_607                  │   ✅   │    NONE    │     -0.56 │     5.80 │ —", "GREEN"),
    ("  FOODS_3_635                  │   ✅   │    NONE    │     -2.81 │     1.85 │ —", "GREEN"),
    ("  FOODS_3_681                  │   ✅   │    NONE    │     +0.72 │     7.11 │ —", "GREEN"),
    ("  FOODS_3_694                  │   ✅   │    NONE    │     -2.19 │     3.60 │ —", "GREEN"),
    ("  FOODS_3_714                  │   ✅   │    NONE    │     -3.10 │     6.20 │ —", "GREEN"),
    ("  FOODS_3_723                  │   ✅   │    NONE    │     -2.76 │     5.99 │ —", "GREEN"),
    ("  FOODS_3_741                  │   ✅   │    NONE    │     -0.74 │     7.45 │ —", "GREEN"),
    ("  FOODS_3_785                  │   ✅   │    NONE    │     -2.41 │    10.82 │ —", "GREEN"),
    ("  FOODS_3_808                  │   ✅   │    NONE    │     -5.15 │     3.87 │ —", "GREEN"),
    ("", "WHITE"),
    ("  ────────────────────────────────────────────────────────────────────", "DIM"),
    ("", "WHITE"),
    ("  SUMMARY", "BOLD"),
    ("     ● Drift detected:   1  SKUs", "RED"),
    ("     ● Warning:          0  SKUs", "YELLOW"),
    ("     ● Stable:          29  SKUs", "GREEN"),
    ("     ──────────────────────────────", "DIM"),
    ("     Total monitored:   30  SKUs", "WHITE"),
]
render_terminal(step4, "step4_drift_detection.png", "Step 4 — Drift Detection", width=2040)


# ═════════════════════════════════════════════════════════════════════
#  STEP 5 — BUSINESS RULES APPLICATION
# ═════════════════════════════════════════════════════════════════════
step5 = [
    ("  [█████] 100% Business Rules", "GREEN"),
    ("", "WHITE"),
    ("════════════════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("  📋  STEP 5 — BUSINESS RULES APPLICATION", "BOLD"),
    ("════════════════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("", "WHITE"),
    ("  Applying 30 rules to 30 SKUs...", "WHITE"),
    ("", "WHITE"),
    ("  SKU                            │  Forecast │  Stock │  Order │ Rule Applied", "DIM"),
    ("  ───────────────────────────────┼───────────┼────────┼────────┼─────────────────────────────", "DIM"),
    ("  FOODS_2_019                    │       100 │     87 │     13 │ Within all constraints", "GREEN"),
    ("  FOODS_2_197                    │       154 │     93 │     61 │ Within all constraints", "GREEN"),
    ("  FOODS_2_276                    │        63 │     76 │      0 │ Sufficient stock", "DIM"),
    ("  FOODS_2_347                    │        78 │     50 │     28 │ Within all constraints", "GREEN"),
    ("  FOODS_3_064                    │       200 │     78 │    104 │ Capped by perishability (7 d)", "YELLOW"),
    ("  FOODS_3_080                    │       128 │     22 │    106 │ Within all constraints", "GREEN"),
    ("  FOODS_3_090                    │       375 │     83 │    292 │ Within all constraints", "GREEN"),
    ("  FOODS_3_099                    │       112 │     17 │     95 │ Within all constraints", "GREEN"),
    ("  FOODS_3_120                    │       284 │     17 │    267 │ Within all constraints", "GREEN"),
    ("  FOODS_3_202                    │       104 │     39 │     65 │ Within all constraints", "GREEN"),
    ("  FOODS_3_252                    │       275 │     49 │    226 │ Within all constraints", "GREEN"),
    ("  FOODS_3_282                    │       138 │     34 │    104 │ Within all constraints", "GREEN"),
    ("  FOODS_3_295                    │       175 │     65 │    103 │ Capped by perishability (7 d)", "YELLOW"),
    ("  FOODS_3_318                    │        89 │     88 │      1 │ Within all constraints", "GREEN"),
    ("  FOODS_3_319                    │         0 │     86 │      0 │ Sufficient stock", "DIM"),
    ("  FOODS_3_491                    │        84 │     11 │     73 │ Within all constraints", "GREEN"),
    ("  FOODS_3_501                    │       171 │     15 │    149 │ Capped by perishability (10 d)", "YELLOW"),
    ("  FOODS_3_541                    │         0 │     61 │      0 │ Sufficient stock", "DIM"),
    ("  FOODS_3_555                    │       152 │     80 │     72 │ Within all constraints", "GREEN"),
    ("  FOODS_3_586                    │       290 │     46 │    244 │ Within all constraints", "GREEN"),
    ("  FOODS_3_587                    │       190 │     32 │    158 │ Within all constraints", "GREEN"),
    ("  FOODS_3_607                    │       123 │     17 │    106 │ Within all constraints", "GREEN"),
    ("  FOODS_3_635                    │         0 │     41 │      0 │ Sufficient stock", "DIM"),
    ("  FOODS_3_681                    │       153 │     45 │    108 │ Within all constraints", "GREEN"),
    ("  FOODS_3_694                    │        68 │     57 │     11 │ Within all constraints", "GREEN"),
    ("  FOODS_3_714                    │       130 │     31 │     99 │ Within all constraints", "GREEN"),
    ("  FOODS_3_723                    │        88 │     58 │     30 │ Within all constraints", "GREEN"),
    ("  FOODS_3_741                    │       107 │     67 │     40 │ Within all constraints", "GREEN"),
    ("  FOODS_3_785                    │       148 │     44 │    104 │ Within all constraints", "GREEN"),
    ("  FOODS_3_808                    │         0 │     47 │      0 │ Sufficient stock", "DIM"),
    ("", "WHITE"),
    ("  ────────────────────────────────────────────────────────────────────", "DIM"),
    ("", "WHITE"),
    ("  SUMMARY", "BOLD"),
    ("     Total forecasted demand:        3,979 units", "CYAN"),
    ("     Total order quantity:           2,659 units", "GREEN"),
    ("     Reduction from rules:            33.2%", "YELLOW"),
    ("     SKUs capped by rules:                3 / 30", "YELLOW"),
]
render_terminal(step5, "step5_business_rules.png", "Step 5 — Business Rules", width=2040)


# ═════════════════════════════════════════════════════════════════════
#  PIPELINE SUMMARY
# ═════════════════════════════════════════════════════════════════════
summary = [
    ("", "WHITE"),
    ("════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("  🏁  PIPELINE COMPLETE", "BOLD"),
    ("════════════════════════════════════════════════════════════════════════", "CYAN"),
    ("", "WHITE"),
    ("  Total time: 3.1s", "WHITE"),
    ("  Data: 56,466 records across 30 SKUs", "CYAN"),
    ("  Store: CA_1", "CYAN"),
    ("  Models trained: 30", "GREEN"),
    ("  Drift events: 1", "YELLOW"),
    ("", "WHITE"),
    ("════════════════════════════════════════════════════════════════════════", "CYAN"),
]
render_terminal(summary, "pipeline_summary.png", "Pipeline Complete", width=1920)


print("\n🎉 All 6 screenshots saved to Images/")
