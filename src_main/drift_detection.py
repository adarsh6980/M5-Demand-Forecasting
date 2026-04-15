"""
Drift Detection Module
Detects concept drift in model predictions using ADWIN and DDM,
both implemented in pure Python (no external drift library needed).
"""
import numpy as np

from scipy import stats
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADWINDetector:
    """
    Simplified ADWIN (Adaptive Windowing) drift detector.
    Compares sliding sub-windows of residuals for statistically
    significant mean shifts.
    """

    def __init__(self, delta=0.002, min_window=10):
        self.delta = delta
        self.min_window = min_window
        self.window = []
        self.drift_detected = False

    def update(self, value):
        """Add a new observation and check for drift."""
        self.window.append(value)
        self.drift_detected = False

        if len(self.window) < self.min_window * 2:
            return False

        for split in range(self.min_window, len(self.window) - self.min_window):
            left = self.window[:split]
            right = self.window[split:]

            mean_diff = abs(np.mean(left) - np.mean(right))
            pooled_std = np.sqrt((np.var(left) + np.var(right)) / 2)

            if pooled_std > 0:
                threshold = np.sqrt(2 * np.log(2 / self.delta) / min(len(left), len(right)))
                if mean_diff / pooled_std > threshold:
                    self.drift_detected = True
                    self.window = self.window[split:]
                    return True

        # Keep the window from growing unbounded
        if len(self.window) > 200:
            self.window = self.window[-100:]

        return False


class DDMDetector:
    """
    Drift Detection Method (DDM).
    Tracks the error rate and its standard deviation; signals drift
    when both exceed historical minimums by configurable thresholds.
    """

    def __init__(self, warning_level=2.0, drift_level=3.0, min_instances=30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_instances = min_instances
        self._reset()

    def update(self, error):
        """Feed a binary error (0 or 1) and check for drift."""
        self.drift_detected = False
        self.warning_detected = False

        self.n += 1
        self.p += (error - self.p) / self.n
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

        if self.p + self.s > self.p_min + self.warning_level * self.s_min:
            self.warning_detected = True

        return False

    def _reset(self):
        self.n = 0
        self.p = 0.0
        self.s = 0.0
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.drift_detected = False
        self.warning_detected = False


class DriftDetector:
    """Runs both ADWIN and DDM on the residual stream for a single SKU."""

    def __init__(self, sku):
        self.sku = sku
        self.adwin = ADWINDetector()
        self.ddm = DDMDetector()
        self.residual_history = []
        self.drift_events = []

    def update(self, residual):
        """
        Feed a new residual (actual − predicted) and return a dict
        describing whether drift was detected and by which method.
        """
        self.residual_history.append(residual)

        self.adwin.update(abs(residual))

        # DDM expects binary errors — threshold against running MAE
        avg_error = np.mean(np.abs(self.residual_history)) if self.residual_history else 0
        binary_error = 1 if abs(residual) > avg_error else 0
        self.ddm.update(binary_error)

        triggered = []
        if self.adwin.drift_detected:
            triggered.append('ADWIN')
        if self.ddm.drift_detected:
            triggered.append('DDM')

        drift_found = len(triggered) > 0
        severity = self._severity() if drift_found else 0.0

        result = {
            'drift_detected': drift_found,
            'detectors_triggered': triggered,
            'severity': severity,
            'message': '',
        }

        if drift_found:
            recent_mae = np.mean(np.abs(self.residual_history[-7:])) if len(self.residual_history) >= 7 else np.mean(np.abs(self.residual_history))
            result['message'] = f"Drift detected by {', '.join(triggered)}. Recent MAE: {recent_mae:.2f}"
            self.drift_events.append({
                'timestamp': datetime.now().isoformat(),
                'sku': self.sku,
                'detectors': triggered,
                'severity': severity,
            })
            logger.warning(f"Drift detected for {self.sku}: {result['message']}")

        return result

    def _severity(self):
        """Score drift severity from 0 to 1 based on recent vs historical MAE."""
        if len(self.residual_history) < 10:
            return 0.5
        recent = self.residual_history[-7:]
        baseline = self.residual_history[:-7] if len(self.residual_history) > 14 else self.residual_history[:7]
        recent_mae = np.mean(np.abs(recent))
        baseline_mae = np.mean(np.abs(baseline)) or recent_mae
        ratio = recent_mae / baseline_mae if baseline_mae else 1.0
        return round(min(1.0, max(0.0, (ratio - 1) / 2)), 2)

    def get_statistics(self):
        """Return current residual summary statistics."""
        if not self.residual_history:
            return {'mean': 0, 'std': 0, 'mae': 0}
        r = np.array(self.residual_history)
        return {
            'mean': float(np.mean(r)),
            'std': float(np.std(r)),
            'mae': float(np.mean(np.abs(r))),
            'count': len(r),
        }


def check_input_drift(feature_before, feature_after, feature_name="feature"):
    """Check for distribution shift using a two-sample KS test."""
    if len(feature_before) < 5 or len(feature_after) < 5:
        return {'drift_detected': False, 'p_value': 1.0, 'message': 'Insufficient data'}

    statistic, p_value = stats.ks_2samp(feature_before, feature_after)
    drifted = p_value < 0.05
    return {
        'feature': feature_name,
        'drift_detected': drifted,
        'ks_statistic': round(statistic, 4),
        'p_value': round(p_value, 4),
        'message': f"{feature_name}: {'Drift detected' if drifted else 'No drift'} (p={p_value:.4f})",
    }


def check_drift(sku_id, residuals, feature_snapshots=None):
    """
    Run a comprehensive drift check for a single SKU by replaying
    its residual history through both detectors.
    """
    detector = DriftDetector(sku_id)

    last_result = None
    for r in residuals:
        last_result = detector.update(r)

    result = {
        'sku': sku_id,
        'drift_detected': last_result['drift_detected'] if last_result else False,
        'severity': last_result['severity'] if last_result else 0.0,
        'detectors_triggered': last_result['detectors_triggered'] if last_result else [],
        'message': last_result['message'] if last_result else 'No drift detected',
        'residual_stats': detector.get_statistics(),
        'input_drift': [],
    }

    if feature_snapshots:
        for name, (before, after) in feature_snapshots.items():
            input_result = check_input_drift(before, after, name)
            result['input_drift'].append(input_result)
            if input_result['drift_detected']:
                result['drift_detected'] = True

    return result


class DriftMonitor:
    """Manages drift detection across all SKUs."""

    def __init__(self):
        self.detectors = {}
        self.events = []

    def get_detector(self, sku):
        """Get or create a detector for a given SKU."""
        if sku not in self.detectors:
            self.detectors[sku] = DriftDetector(sku)
        return self.detectors[sku]

    def update(self, sku, actual, predicted):
        """Feed a new (actual, predicted) pair and check for drift."""
        residual = actual - predicted
        detector = self.get_detector(sku)
        result = detector.update(residual)

        if result['drift_detected']:
            self.events.append({
                'timestamp': datetime.now().isoformat(),
                'sku': sku,
                'detectors': result['detectors_triggered'],
                'severity': result['severity'],
                'message': result['message'],
            })
        return result

    def get_all_events(self):
        return self.events

    def get_sku_events(self, sku):
        return [e for e in self.events if e['sku'] == sku]


if __name__ == "__main__":
    np.random.seed(42)
    normal = np.random.normal(0, 5, 30).tolist()
    shifted = np.random.normal(10, 8, 20).tolist()

    result = check_drift("MILK_1L", normal + shifted)
    print(f"\nDrift Check — MILK_1L:")
    print(f"  Detected: {result['drift_detected']}")
    print(f"  Severity: {result['severity']}")
    print(f"  Detectors: {result['detectors_triggered']}")
    print(f"  Message: {result['message']}")
