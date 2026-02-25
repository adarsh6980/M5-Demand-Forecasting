"""
Drift Detection Module
Detects concept drift in model predictions using statistical methods.
Provides pure Python implementation (no external drift detection library required).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging
from datetime import datetime
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADWINDetector:
    """
    Pure Python implementation of ADWIN (Adaptive Windowing) drift detector.
    Simplified version using sliding window statistics comparison.
    """
    
    def __init__(self, delta: float = 0.002, min_window: int = 10):
        self.delta = delta
        self.min_window = min_window
        self.window: List[float] = []
        self.drift_detected = False
        
    def update(self, value: float) -> bool:
        """Add new value and check for drift."""
        self.window.append(value)
        self.drift_detected = False
        
        if len(self.window) < self.min_window * 2:
            return False
        
        # Check for drift by comparing subwindows
        for split in range(self.min_window, len(self.window) - self.min_window):
            left = self.window[:split]
            right = self.window[split:]
            
            mean_diff = abs(np.mean(left) - np.mean(right))
            pooled_std = np.sqrt((np.var(left) + np.var(right)) / 2)
            
            if pooled_std > 0:
                # Statistical threshold based on window sizes
                threshold = np.sqrt(2 * np.log(2 / self.delta) / min(len(left), len(right)))
                if mean_diff / pooled_std > threshold:
                    self.drift_detected = True
                    # Shrink window to recent data
                    self.window = self.window[split:]
                    return True
        
        # Limit window size
        if len(self.window) > 200:
            self.window = self.window[-100:]
        
        return False


class DDMDetector:
    """
    Pure Python implementation of DDM (Drift Detection Method).
    """
    
    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0, min_instances: int = 30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_instances = min_instances
        
        self.n = 0
        self.p = 0.0
        self.s = 0.0
        self.p_min = float('inf')
        self.s_min = float('inf')
        
        self.drift_detected = False
        self.warning_detected = False
        
    def update(self, error: int) -> bool:
        """Update with binary error (0 or 1) and check for drift."""
        self.drift_detected = False
        self.warning_detected = False
        
        self.n += 1
        self.p = self.p + (error - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n)
        
        if self.n < self.min_instances:
            return False
        
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        
        if self.p + self.s > self.p_min + self.drift_level * self.s_min:
            self.drift_detected = True
            self._reset()
            return True
        elif self.p + self.s > self.p_min + self.warning_level * self.s_min:
            self.warning_detected = True
        
        return False
    
    def _reset(self):
        """Reset detector after drift."""
        self.n = 0
        self.p = 0.0
        self.s = 0.0
        self.p_min = float('inf')
        self.s_min = float('inf')


class DriftDetector:
    """Manages drift detection for a single SKU using multiple methods."""
    
    def __init__(self, sku: str):
        self.sku = sku
        self.adwin = ADWINDetector()
        self.ddm = DDMDetector()
        self.residual_history: List[float] = []
        self.drift_events: List[Dict] = []
        
    def update(self, residual: float) -> Dict:
        """
        Update detectors with new residual and check for drift.
        
        Args:
            residual: Prediction error (actual - predicted)
            
        Returns:
            Dict with drift detection results
        """
        self.residual_history.append(residual)
        
        # Update ADWIN with absolute residual
        self.adwin.update(abs(residual))
        
        # DDM expects binary classification errors, use thresholded residual
        threshold = np.mean(np.abs(self.residual_history)) if self.residual_history else 0
        binary_error = 1 if abs(residual) > threshold else 0
        self.ddm.update(binary_error)
        
        result = {
            'drift_detected': False,
            'detectors_triggered': [],
            'severity': 0.0,
            'message': ''
        }
        
        if self.adwin.drift_detected:
            result['drift_detected'] = True
            result['detectors_triggered'].append('ADWIN')
            
        if self.ddm.drift_detected:
            result['drift_detected'] = True
            result['detectors_triggered'].append('DDM')
        
        if result['drift_detected']:
            result['severity'] = self._calculate_severity()
            result['message'] = self._generate_message(result['detectors_triggered'])
            self.drift_events.append({
                'timestamp': datetime.now().isoformat(),
                'sku': self.sku,
                'detectors': result['detectors_triggered'],
                'severity': result['severity']
            })
            logger.warning(f"Drift detected for {self.sku}: {result['message']}")
        
        return result
    
    def _calculate_severity(self) -> float:
        """Calculate drift severity based on residual statistics."""
        if len(self.residual_history) < 10:
            return 0.5
        
        recent = self.residual_history[-7:]
        historical = self.residual_history[:-7] if len(self.residual_history) > 14 else self.residual_history[:7]
        
        recent_mae = np.mean(np.abs(recent))
        historical_mae = np.mean(np.abs(historical)) if historical else recent_mae
        
        if historical_mae == 0:
            return 0.5
        
        ratio = recent_mae / historical_mae
        severity = min(1.0, max(0.0, (ratio - 1) / 2))
        return round(severity, 2)
    
    def _generate_message(self, detectors: List[str]) -> str:
        """Generate human-readable drift message."""
        detector_str = ', '.join(detectors)
        recent_residuals = self.residual_history[-7:] if len(self.residual_history) >= 7 else self.residual_history
        recent_mae = np.mean(np.abs(recent_residuals))
        return f"Drift detected by {detector_str}. Recent MAE: {recent_mae:.2f}"
    
    def get_statistics(self) -> Dict:
        """Get current residual statistics."""
        if not self.residual_history:
            return {'mean': 0, 'std': 0, 'mae': 0}
        
        residuals = np.array(self.residual_history)
        return {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'mae': float(np.mean(np.abs(residuals))),
            'count': len(residuals)
        }


def check_input_drift(feature_before: np.ndarray, feature_after: np.ndarray, 
                      feature_name: str = "feature") -> Dict:
    """
    Check for input distribution drift using KS-test.
    """
    if len(feature_before) < 5 or len(feature_after) < 5:
        return {'drift_detected': False, 'p_value': 1.0, 'message': 'Insufficient data'}
    
    statistic, p_value = stats.ks_2samp(feature_before, feature_after)
    drift_detected = p_value < 0.05
    
    return {
        'feature': feature_name,
        'drift_detected': drift_detected,
        'ks_statistic': round(statistic, 4),
        'p_value': round(p_value, 4),
        'message': f"{feature_name}: {'Drift detected' if drift_detected else 'No drift'} (p={p_value:.4f})"
    }


def check_drift(sku_id: str, residual_series: List[float], 
                feature_snapshots: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None) -> Dict:
    """
    Comprehensive drift check for a SKU.
    """
    detector = DriftDetector(sku_id)
    
    final_result = None
    for residual in residual_series:
        final_result = detector.update(residual)
    
    result = {
        'sku': sku_id,
        'drift_detected': final_result['drift_detected'] if final_result else False,
        'severity': final_result['severity'] if final_result else 0.0,
        'detectors_triggered': final_result['detectors_triggered'] if final_result else [],
        'message': final_result['message'] if final_result else 'No drift detected',
        'residual_stats': detector.get_statistics(),
        'input_drift': []
    }
    
    if feature_snapshots:
        for feature_name, (before, after) in feature_snapshots.items():
            input_result = check_input_drift(before, after, feature_name)
            result['input_drift'].append(input_result)
            if input_result['drift_detected']:
                result['drift_detected'] = True
    
    return result


class DriftMonitor:
    """Manages drift detection across all SKUs."""
    
    def __init__(self):
        self.detectors: Dict[str, DriftDetector] = {}
        self.events: List[Dict] = []
        
    def get_detector(self, sku: str) -> DriftDetector:
        """Get or create detector for SKU."""
        if sku not in self.detectors:
            self.detectors[sku] = DriftDetector(sku)
        return self.detectors[sku]
    
    def update(self, sku: str, actual: float, predicted: float) -> Dict:
        """Update drift detector with new prediction result."""
        residual = actual - predicted
        detector = self.get_detector(sku)
        result = detector.update(residual)
        
        if result['drift_detected']:
            event = {
                'timestamp': datetime.now().isoformat(),
                'sku': sku,
                'detectors': result['detectors_triggered'],
                'severity': result['severity'],
                'message': result['message']
            }
            self.events.append(event)
        
        return result
    
    def get_all_events(self) -> List[Dict]:
        """Get all drift events."""
        return self.events
    
    def get_sku_events(self, sku: str) -> List[Dict]:
        """Get drift events for specific SKU."""
        return [e for e in self.events if e['sku'] == sku]


if __name__ == "__main__":
    # Test drift detection
    np.random.seed(42)
    
    # Simulate residuals with drift
    normal_residuals = np.random.normal(0, 5, 30).tolist()
    drift_residuals = np.random.normal(10, 8, 20).tolist()
    all_residuals = normal_residuals + drift_residuals
    
    result = check_drift("MILK_1L", all_residuals)
    print(f"\nDrift Check Results for MILK_1L:")
    print(f"  Drift Detected: {result['drift_detected']}")
    print(f"  Severity: {result['severity']}")
    print(f"  Detectors: {result['detectors_triggered']}")
    print(f"  Message: {result['message']}")
