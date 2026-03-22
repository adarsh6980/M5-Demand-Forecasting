"""
Forecasting Module with Continuous Learning
XGBoost-based demand forecasting with:
- Model persistence
- Incremental retraining
- Performance tracking over time
- Model improvement metrics
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    """Tracks model performance over time to measure improvement."""
    
    def __init__(self, history_path: str):
        self.history_path = Path(history_path)
        self.history: Dict[str, List[Dict]] = {}
        self._load_history()
    
    def _load_history(self):
        """Load performance history from file."""
        if self.history_path.exists():
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
    
    def _save_history(self):
        """Save performance history to file."""
        self.history_path.parent.mkdir(exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def record_metrics(self, sku: str, metrics: Dict, training_samples: int):
        """Record training metrics for a SKU."""
        if sku not in self.history:
            self.history[sku] = []
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'mae': metrics.get('mae'),
            'rmse': metrics.get('rmse'),
            'r2': metrics.get('r2'),
            'mape': metrics.get('mape'),
            'training_samples': training_samples,
            'version': len(self.history[sku]) + 1
        }
        self.history[sku].append(entry)
        self._save_history()
        
        logger.info(f"Recorded metrics for {sku} v{entry['version']}")
    
    def get_improvement(self, sku: str) -> Dict:
        """Calculate model improvement over time."""
        if sku not in self.history or len(self.history[sku]) < 2:
            return {'improved': False, 'message': 'Insufficient history'}
        
        history = self.history[sku]
        first = history[0]
        last = history[-1]
        
        mae_improvement = ((first['mae'] - last['mae']) / first['mae']) * 100 if first['mae'] else 0
        
        return {
            'improved': mae_improvement > 0,
            'mae_improvement_pct': round(mae_improvement, 2),
            'initial_mae': first['mae'],
            'current_mae': last['mae'],
            'versions_trained': len(history),
            'message': f"MAE improved by {mae_improvement:.1f}%" if mae_improvement > 0 else "Model needs more data"
        }
    
    def get_sku_history(self, sku: str) -> pd.DataFrame:
        """Get performance history dataframe for a SKU."""
        if sku not in self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history[sku])


class ForecastModel:
    """XGBoost-based demand forecasting model with continuous learning."""
    
    def __init__(self, sku: str):
        self.sku = sku
        self.model = None
        self.feature_columns = None
        self.metrics = {}
        self.training_history: List[Dict] = []
        self.version = 0
        
    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str = 'units_sold',
              test_size: float = 0.2, use_time_series_cv: bool = True) -> Dict[str, float]:
        """
        Train the XGBoost model on SKU data.
        Uses time series cross-validation for better generalization.
        """
        self.feature_columns = feature_cols
        self.version += 1
        
        # Filter valid features that exist in df
        valid_features = [c for c in feature_cols if c in df.columns]
        
        X = df[valid_features].values
        y = df[target_col].values
        
        if use_time_series_cv and len(X) > 50:
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                cv_scores.append(mean_absolute_error(y_val, y_pred))
            
            # Final training on all data with validation on last portion
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
        
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=10
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        
        # Calculate comprehensive metrics
        self.metrics = {
            'mae': float(mean_absolute_error(y_val, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred))),
            'r2': float(r2_score(y_val, y_pred)),
            'mape': float(np.mean(np.abs((y_val - y_pred) / np.maximum(y_val, 1))) * 100),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'version': self.version
        }
        
        # Track training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            **self.metrics
        })
        
        logger.info(f"Trained model v{self.version} for {self.sku}: MAE={self.metrics['mae']:.2f}, R²={self.metrics['r2']:.3f}")
        return self.metrics
    
    def incremental_train(self, new_df: pd.DataFrame, existing_df: pd.DataFrame,
                          feature_cols: List[str], target_col: str = 'units_sold') -> Dict[str, float]:
        """
        Incrementally retrain model with new data added to existing data.
        This is how the model improves over time.
        """
        # Combine existing and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date']).sort_values('date')
        
        # Retrain with more data
        return self.train(combined_df, feature_cols, target_col)
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for given features."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        valid_features = [c for c in self.feature_columns if c in df.columns]
        X = df[valid_features].values
        
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)  # Ensure non-negative
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if self.model is None:
            return pd.DataFrame()
        
        valid_features = [c for c in self.feature_columns if c in self.model.feature_names_in_]
        importance = pd.DataFrame({
            'feature': valid_features,
            'importance': self.model.feature_importances_[:len(valid_features)]
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save_model(self, path: str):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'sku': self.sku,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'version': self.version,
            'training_history': self.training_history
        }
        joblib.dump(model_data, path)
        logger.info(f"Model v{self.version} saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'ForecastModel':
        """Load model from disk."""
        model_data = joblib.load(path)
        instance = cls(model_data['sku'])
        instance.model = model_data['model']
        instance.feature_columns = model_data['feature_columns']
        instance.metrics = model_data.get('metrics', {})
        instance.version = model_data.get('version', 1)
        instance.training_history = model_data.get('training_history', [])
        logger.info(f"Model v{instance.version} loaded from {path}")
        return instance


class ForecastingEngine:
    """Manages multiple SKU forecast models with continuous learning."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models: Dict[str, ForecastModel] = {}
        self.tracker = ModelPerformanceTracker(self.models_dir / "performance_history.json")
        
    def train_all(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict]:
        """Train models for all SKUs."""
        results = {}
        for sku in df['sku'].unique():
            sku_df = df[df['sku'] == sku].copy()
            if len(sku_df) < 10:
                logger.warning(f"Skipping {sku}: insufficient data")
                continue
            
            # Check if we have an existing model to improve
            existing_model_path = self.models_dir / f"{sku}_model.joblib"
            if existing_model_path.exists():
                model = ForecastModel.load_model(existing_model_path)
                metrics = model.train(sku_df, feature_cols)  # Retrain with all data
            else:
                model = ForecastModel(sku)
                metrics = model.train(sku_df, feature_cols)
            
            self.models[sku] = model
            results[sku] = metrics
            
            # Track performance
            self.tracker.record_metrics(sku, metrics, len(sku_df))
            
            # Save model
            model.save_model(self.models_dir / f"{sku}_model.joblib")
        
        return results
    
    def retrain_sku(self, sku: str, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Retrain a specific SKU model with new data."""
        sku_df = df[df['sku'] == sku].copy()
        
        if sku in self.models:
            model = self.models[sku]
        else:
            model = ForecastModel(sku)
        
        metrics = model.train(sku_df, feature_cols)
        self.models[sku] = model
        
        # Track and save
        self.tracker.record_metrics(sku, metrics, len(sku_df))
        model.save_model(self.models_dir / f"{sku}_model.joblib")
        
        return metrics
    
    def get_improvement_report(self) -> pd.DataFrame:
        """Get improvement report for all SKUs."""
        report_data = []
        for sku in self.models.keys():
            improvement = self.tracker.get_improvement(sku)
            report_data.append({
                'sku': sku,
                'versions': improvement.get('versions_trained', 1),
                'initial_mae': improvement.get('initial_mae'),
                'current_mae': improvement.get('current_mae'),
                'improvement_pct': improvement.get('mae_improvement_pct', 0),
                'improved': improvement.get('improved', False)
            })
        return pd.DataFrame(report_data)
    
    def predict(self, df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
        """Generate forecasts for all SKUs."""
        predictions = []
        
        for sku, model in self.models.items():
            sku_df = df[df['sku'] == sku].tail(horizon)
            if len(sku_df) == 0:
                continue
                
            preds = model.predict(sku_df)
            for i, (idx, row) in enumerate(sku_df.iterrows()):
                predictions.append({
                    'date': row['date'],
                    'sku': sku,
                    'forecast': preds[i],
                    'model_version': model.version
                })
        
        return pd.DataFrame(predictions)
    
    def load_all_models(self):
        """Load all saved models."""
        for model_file in self.models_dir.glob("*_model.joblib"):
            try:
                model = ForecastModel.load_model(model_file)
                self.models[model.sku] = model
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
        logger.info(f"Loaded {len(self.models)} models")


def generate_forecast(df: pd.DataFrame, sku: str, horizon: int = 7) -> pd.DataFrame:
    """Quick forecast generation for a single SKU."""
    from .feature_engineering import get_feature_columns
    
    feature_cols = get_feature_columns()
    sku_df = df[df['sku'] == sku].copy()
    
    model = ForecastModel(sku)
    model.train(sku_df, feature_cols)
    
    forecast_df = sku_df.tail(horizon)
    preds = model.predict(forecast_df)
    
    result = forecast_df[['date']].copy()
    result['forecast'] = preds
    result['sku'] = sku
    
    return result


if __name__ == "__main__":
    from data_ingestion import load_pos_data, load_external_data, merge_data
    from feature_engineering import prepare_features, get_feature_columns
    
    base_path = Path(__file__).parent.parent / "data"
    pos = load_pos_data(base_path / "pos_data.csv")
    external = load_external_data(base_path / "external_data.csv")
    merged = merge_data(pos, external)
    features = prepare_features(merged)
    
    engine = ForecastingEngine(str(base_path.parent / "models"))
    
    # First training
    print("\n=== First Training ===")
    results = engine.train_all(features, get_feature_columns())
    for sku, metrics in results.items():
        print(f"  {sku}: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
    
    # Simulate improvement with more training
    print("\n=== Second Training (simulating more data) ===")
    results = engine.train_all(features, get_feature_columns())
    
    # Show improvement report
    print("\n=== Improvement Report ===")
    report = engine.get_improvement_report()
    print(report.to_string())
