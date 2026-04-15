"""
Feature Engineering Module
Creates calendar, lag, and rolling features for demand forecasting
from the merged M5 dataset.
"""
import pandas as pd
import numpy as np
import holidays
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-build the US holiday calendar once at import time
_US_HOLIDAYS = holidays.US()


def add_calendar_features(df):
    """Derive day-of-week, month, weekend, and holiday flags from the date column."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['date'].isin(_US_HOLIDAYS).astype(int)
    logger.info("Added calendar features")
    return df


def add_lag_features(df, lags=None):
    """Shift units_sold by the given lag periods, grouped by SKU."""
    if lags is None:
        lags = [1, 7, 14]
    df = df.copy().sort_values(['sku', 'date'])
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('sku')['units_sold'].shift(lag)
    logger.info(f"Added lag features: {lags}")
    return df


def add_rolling_features(df, windows=None):
    """Compute rolling mean and std of units_sold, grouped by SKU."""
    if windows is None:
        windows = [7, 14]
    df = df.copy().sort_values(['sku', 'date'])
    for w in windows:
        df[f'rolling_mean_{w}'] = df.groupby('sku')['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f'rolling_std_{w}'] = df.groupby('sku')['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).std()
        )
    logger.info(f"Added rolling features for windows: {windows}")
    return df


def prepare_features(df, include_lags=True):
    """Run the full feature-preparation pipeline."""
    df = add_calendar_features(df)
    if include_lags:
        df = add_lag_features(df)
        df = add_rolling_features(df)
    df = df.fillna(0)
    logger.info(f"Feature preparation complete. Shape: {df.shape}")
    return df


def get_feature_columns():
    """Return the fixed list of feature column names used for modelling."""
    return [
        'day_of_week', 'day_of_month', 'month', 'week_of_year',
        'is_weekend', 'is_holiday', 'promo_flag',
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_std_7',
        'rolling_mean_14', 'rolling_std_14',
    ]


if __name__ == "__main__":
    from data_ingestion import load_calendar, load_sales, load_prices, melt_and_merge

    cal = load_calendar()
    sales = load_sales()
    prices = load_prices()
    merged = melt_and_merge(sales, cal, prices)
    features = prepare_features(merged)
    print(features.head())
    print(f"\nFeature columns: {get_feature_columns()}")
