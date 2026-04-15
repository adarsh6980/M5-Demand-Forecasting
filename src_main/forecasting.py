"""
Forecasting Module
XGBoost-based demand forecasting with model persistence,
performance tracking, and incremental retraining support.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from datetime import datetime
import joblib
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPerformanceTracker:
    """Persists and compares model metrics across training runs."""

    def __init__(self, history_path):
        self.history_path = Path(history_path)
        self.history = {}
        if self.history_path.exists():
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)

    def _save(self):
        self.history_path.parent.mkdir(exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def record_metrics(self, sku, metrics, training_samples):
        """Append a metrics snapshot for a SKU and persist to disk."""
        if sku not in self.history:
            self.history[sku] = []

        entry = {
            'timestamp': datetime.now().isoformat(),
            'mae': metrics.get('mae'),
            'rmse': metrics.get('rmse'),
            'r2': metrics.get('r2'),
            'mape': metrics.get('mape'),
            'training_samples': training_samples,
            'version': len(self.history[sku]) + 1,
        }
        self.history[sku].append(entry)
        self._save()
        logger.info(f"Recorded metrics for {sku} v{entry['version']}")

    def get_improvement(self, sku):
        """Compare first and latest training run for a SKU."""
        entries = self.history.get(sku, [])
        if len(entries) < 2:
            return {'improved': False, 'message': 'Insufficient history'}

        first, last = entries[0], entries[-1]
        pct = ((first['mae'] - last['mae']) / first['mae'] * 100) if first['mae'] else 0

        return {
            'improved': pct > 0,
            'mae_improvement_pct': round(pct, 2),
            'initial_mae': first['mae'],
            'current_mae': last['mae'],
            'versions_trained': len(entries),
            'message': f"MAE improved by {pct:.1f}%" if pct > 0 else "Model needs more data",
        }

    def get_sku_history(self, sku):
        return pd.DataFrame(self.history.get(sku, []))


class ForecastModel:
    """Per-SKU XGBoost model with train / predict / save / load lifecycle."""

    def __init__(self, sku):
        self.sku = sku
        self.model = None
        self.feature_columns = None
        self.metrics = {}
        self.training_history = []
        self.version = 0

    def train(self, df, feature_cols, target_col='units_sold',
              test_size=0.2, use_time_series_cv=True):
        """
        Train on the provided DataFrame.  Optionally runs 3-fold
        time-series CV before the final fit with early stopping.
        """
        self.feature_columns = feature_cols
        self.version += 1

        valid_features = [col for col in feature_cols if col in df.columns]
        X = df[valid_features].values
        y = df[target_col].values

        # Optional cross-validation pass (scores are logged but the
        # final model is always trained on the full split below)
        if use_time_series_cv and len(X) > 50:
            tscv = TimeSeriesSplit(n_splits=3)
            for train_idx, val_idx in tscv.split(X):
                fold_model = XGBRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    random_state=42, verbosity=0,
                )
                fold_model.fit(X[train_idx], y[train_idx])

        # Final train/val split (chronological, no shuffle)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        self.model = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=0,
            early_stopping_rounds=10, tree_method='hist', n_jobs=1,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = self.model.predict(X_val)

        self.metrics = {
            'mae': float(mean_absolute_error(y_val, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred))),
            'r2': float(r2_score(y_val, y_pred)),
            'mape': float(np.mean(np.abs((y_val - y_pred) / np.maximum(y_val, 1))) * 100),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'version': self.version,
        }
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            **self.metrics,
        })
        logger.info(f"Trained v{self.version} for {self.sku}: MAE={self.metrics['mae']:.2f}, R²={self.metrics['r2']:.3f}")
        return self.metrics

    def incremental_train(self, new_df, existing_df, feature_cols,
                          target_col='units_sold'):
        """Retrain on the union of old + new data (deduped by date)."""
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date']).sort_values('date')
        return self.train(combined, feature_cols, target_col)

    def predict(self, df):
        """Return non-negative predictions for the given feature DataFrame."""
        if self.model is None:
            raise ValueError("Model not trained — call train() first.")
        valid_features = [col for col in self.feature_columns if col in df.columns]
        return np.maximum(self.model.predict(df[valid_features].values), 0)

    def get_feature_importance(self):
        """Return a sorted DataFrame of feature importances."""
        if self.model is None:
            return pd.DataFrame()
        names = list(self.model.feature_names_in_)
        return pd.DataFrame({
            'feature': names,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)

    def save_model(self, path):
        joblib.dump({
            'model': self.model,
            'sku': self.sku,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'version': self.version,
            'training_history': self.training_history,
        }, path)
        logger.info(f"Model v{self.version} saved to {path}")

    @classmethod
    def load_model(cls, path):
        data = joblib.load(path)
        inst = cls(data['sku'])
        inst.model = data['model']
        inst.feature_columns = data['feature_columns']
        inst.metrics = data.get('metrics', {})
        inst.version = data.get('version', 1)
        inst.training_history = data.get('training_history', [])
        logger.info(f"Model v{inst.version} loaded from {path}")
        return inst


class ForecastingEngine:
    """Manages a portfolio of per-SKU forecast models."""

    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models = {}
        self.tracker = ModelPerformanceTracker(self.models_dir / "performance_history.json")

    def train_all(self, df, feature_cols):
        """Train (or retrain) models for every SKU in the DataFrame."""
        results = {}
        for sku in df['sku'].unique():
            sku_df = df[df['sku'] == sku].copy()
            if len(sku_df) < 10:
                logger.warning(f"Skipping {sku}: insufficient data")
                continue

            model_path = self.models_dir / f"{sku}_model.joblib"
            if model_path.exists():
                model = ForecastModel.load_model(model_path)
                metrics = model.train(sku_df, feature_cols)
            else:
                model = ForecastModel(sku)
                metrics = model.train(sku_df, feature_cols)

            self.models[sku] = model
            results[sku] = metrics
            self.tracker.record_metrics(sku, metrics, len(sku_df))
            model.save_model(model_path)

        return results

    def retrain_sku(self, sku, df, feature_cols):
        """Retrain a single SKU model."""
        sku_df = df[df['sku'] == sku].copy()
        model = self.models.get(sku) or ForecastModel(sku)
        metrics = model.train(sku_df, feature_cols)
        self.models[sku] = model
        self.tracker.record_metrics(sku, metrics, len(sku_df))
        model.save_model(self.models_dir / f"{sku}_model.joblib")
        return metrics

    def get_improvement_report(self):
        """Build a DataFrame comparing initial vs current MAE per SKU."""
        rows = []
        for sku in self.models:
            imp = self.tracker.get_improvement(sku)
            rows.append({
                'sku': sku,
                'versions': imp.get('versions_trained', 1),
                'initial_mae': imp.get('initial_mae'),
                'current_mae': imp.get('current_mae'),
                'improvement_pct': imp.get('mae_improvement_pct', 0),
                'improved': imp.get('improved', False),
            })
        return pd.DataFrame(rows)

    def predict(self, df, horizon=7):
        """Generate forecasts for all loaded SKU models."""
        rows = []
        for sku, model in self.models.items():
            sku_df = df[df['sku'] == sku].tail(horizon)
            if sku_df.empty:
                continue
            preds = model.predict(sku_df)
            for i, (_, row) in enumerate(sku_df.iterrows()):
                rows.append({
                    'date': row['date'],
                    'sku': sku,
                    'forecast': preds[i],
                    'model_version': model.version,
                })
        return pd.DataFrame(rows)

    def load_all_models(self):
        """Load every saved model from the models directory."""
        for path in self.models_dir.glob("*_model.joblib"):
            try:
                model = ForecastModel.load_model(path)
                self.models[model.sku] = model
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        logger.info(f"Loaded {len(self.models)} models")


def generate_forecast(df, sku, horizon=7):
    """Quick one-shot forecast for a single SKU."""
    from feature_engineering import get_feature_columns

    feature_cols = get_feature_columns()
    sku_df = df[df['sku'] == sku].copy()

    model = ForecastModel(sku)
    model.train(sku_df, feature_cols)

    tail = sku_df.tail(horizon)
    result = tail[['date']].copy()
    result['forecast'] = model.predict(tail)
    result['sku'] = sku
    return result


if __name__ == "__main__":
    from data_ingestion import load_calendar, load_sales, load_prices, melt_and_merge
    from feature_engineering import prepare_features, get_feature_columns

    cal = load_calendar()
    sales = load_sales()
    prices = load_prices()
    merged = melt_and_merge(sales, cal, prices)
    features = prepare_features(merged)

    engine = ForecastingEngine(str(Path(__file__).parent.parent / "models1"))

    print("\n=== Training ===")
    results = engine.train_all(features, get_feature_columns())
    for sku, m in results.items():
        print(f"  {sku}: MAE={m['mae']:.2f}, R²={m['r2']:.3f}")
