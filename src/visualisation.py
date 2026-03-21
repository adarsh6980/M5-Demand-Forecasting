"""
M5 Demand Forecasting — Visualization Module
Generates research-quality exploratory visualisations
for adaptive demand forecasting insights.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Professional styling
sns.set_theme(style="whitegrid", context="talk")

OUTPUT_PATH = Path(__file__).parent.parent / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)


# ============================================================
# 1️⃣ WEEKLY SALES TREND (reduces noise, highlights seasonality)
# ============================================================

def plot_sales_trend(df):

    weekly_sales = df.resample("W", on="date")["units_sold"].sum()

    plt.figure(figsize=(12, 5))

    sns.lineplot(
        x=weekly_sales.index,
        y=weekly_sales.values,
        linewidth=2.5,
        color="#2a9d8f"
    )

    plt.title("Weekly Retail Demand Trend (Store-Level)")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "sales_trend.png")
    plt.close()


# ============================================================
# 2️⃣ PROMOTION UPLIFT ANALYSIS
# ============================================================

def plot_promo_effect(df):

    promo_sales = df.groupby("promo_flag")["units_sold"].mean().reset_index()

    promo_sales["promo_label"] = promo_sales["promo_flag"].map({
        0: "No Promotion",
        1: "Promotion Active"
    })

    plt.figure(figsize=(6,5))

    sns.barplot(
        data=promo_sales,
        x="promo_label",
        y="units_sold",
        hue="promo_label",
        palette=["#457b9d", "#e63946"],
        legend=False
    )

    plt.title("Promotion Impact on Demand")
    plt.ylabel("Average Units Sold")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "promo_effect.png")
    plt.close()


# ============================================================
# 3️⃣ PRICE ELASTICITY ANALYSIS (BINNED PRICE RESPONSE CURVE)
# ============================================================

def plot_price_vs_sales(df):

    price_response = df.groupby("price")["units_sold"].mean().reset_index()

    plt.figure(figsize=(8, 5))

    sns.lineplot(
        data=price_response,
        x="price",
        y="units_sold",
        marker="o",
        linewidth=2,
        color="#264653"
    )

    plt.title("Average Demand Response Across Price Levels")
    plt.xlabel("Price")
    plt.ylabel("Average Units Sold")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "price_vs_demand.png")
    plt.close()


# ============================================================
# 4️⃣ CATEGORY-LEVEL DEMAND STRUCTURE (M5 CORE INSIGHT)
# ============================================================

def plot_category_sales(df):

    cat_sales = df.groupby("cat_id")["units_sold"].mean().reset_index()

    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=cat_sales,
        x="cat_id",
        y="units_sold",
        palette="viridis"
    )

    plt.title("Average Demand by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Average Units Sold")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "category_sales.png")
    plt.close()


# ============================================================
# RUN STANDALONE (FOR TESTING)
# ============================================================

if __name__ == "__main__":

    BASE_PATH = Path(__file__).parent.parent
    DATA_PATH = BASE_PATH / "data" / "pos_data.csv"

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    print("Generating research-grade visualisations...")

    plot_sales_trend(df)
    plot_promo_effect(df)
    plot_price_vs_sales(df)
    plot_category_sales(df)

    print("✅ Visualisations saved successfully in outputs/")