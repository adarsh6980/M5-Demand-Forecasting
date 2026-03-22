"""
Feature Engineering Module — Raw M5 Data Pipeline
Creates calendar, lag, and rolling features for demand forecasting.
Same features as the original pipeline but with source-awareness for 3-CSV data.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import holidays
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features to the dataframe."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # US holidays
    us_holidays = holidays.US()
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in us_holidays else 0)

    logger.info("Added calendar features")
    return df


def add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 7, 14]) -> pd.DataFrame:
    """Add lag features for time series forecasting."""
    df = df.copy()
    df = df.sort_values(['sku', 'date'])

    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('sku')['units_sold'].shift(lag)

    logger.info(f"Added lag features: {lags}")
    return df


def add_rolling_features(df: pd.DataFrame, windows: List[int] = [7, 14]) -> pd.DataFrame:
    """Add rolling statistics features."""
    df = df.copy()
    df = df.sort_values(['sku', 'date'])

    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby('sku')['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby('sku')['units_sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )

    logger.info(f"Added rolling features for windows: {windows}")
    return df


def prepare_features(df: pd.DataFrame, include_lags: bool = True) -> pd.DataFrame:
    """Full feature preparation pipeline."""
    df = add_calendar_features(df)

    if include_lags:
        df = add_lag_features(df)
        df = add_rolling_features(df)

    # Fill NaN values from lag/rolling with 0
    df = df.fillna(0)

    logger.info(f"Feature preparation complete. Shape: {df.shape}")
    return df


def get_feature_columns() -> List[str]:
    """Return list of feature column names used for modeling."""
    return [
        'day_of_week', 'day_of_month', 'month', 'week_of_year',
        'is_weekend', 'is_holiday', 'promo_flag',
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_std_7',
        'rolling_mean_14', 'rolling_std_14'
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
