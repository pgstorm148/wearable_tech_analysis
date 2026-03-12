import numpy as np
from scipy import signal
from typing import Tuple

class KalmanFilter1D:
    """Scalar Kalman filter for smoothing noisy signals."""
    def __init__(self, process_noise: float = 1e-5, measurement_noise: float = 1e-2):
        self.Q = process_noise
        self.R = measurement_noise
        self.reset()
        
    def reset(self):
        self.x = 0.0
        self.P = 1.0
        
    def update(self, measurement: float) -> float:
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x

class FFTBandpassFilter:
    """Applies a Butterworth bandpass filter and optional notch filter."""
    def __init__(self, lowcut: float, highcut: float, fs: float, order: int = 4, notch_freq: float = None):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.notch_freq = notch_freq
        
    def filter(self, data: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        if len(data) < 10:
            return data, 0.0, np.array([])
            
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = signal.butter(self.order, [low, high], btype='band')
        
        padlen = min(len(data) - 1, 3 * max(len(a), len(b)))
        if len(data) > padlen:
            filtered = signal.filtfilt(b, a, data, padlen=padlen)
        else:
            filtered = signal.lfilter(b, a, data)
        
        if self.notch_freq:
            w0 = self.notch_freq / nyq
            b_notch, a_notch = signal.iirnotch(w0, 30.0)
            if len(filtered) > padlen:
                filtered = signal.filtfilt(b_notch, a_notch, filtered, padlen=padlen)
            else:
                filtered = signal.lfilter(b_notch, a_notch, filtered)
            
        freqs = np.fft.rfftfreq(len(filtered), 1/self.fs)
        fft_vals = np.abs(np.fft.rfft(filtered))
        power_spectrum = fft_vals ** 2
        
        if len(power_spectrum) > 0:
            dominant_frequency = freqs[np.argmax(power_spectrum)]
        else:
            dominant_frequency = 0.0
            
        return filtered, dominant_frequency, power_spectrum

class MotionArtifactRemover:
    """Removes motion artifacts using Z-score thresholding and interpolation."""
    def __init__(self, threshold_z: float = 3.0):
        self.threshold_z = threshold_z
        
    def process(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return data.copy(), np.zeros_like(data, dtype=bool)
            
        z_scores = np.abs((data - mean_val) / std_val)
        mask = z_scores > self.threshold_z
        
        cleaned = data.copy()
        if np.any(mask):
            x = np.arange(len(data))
            cleaned[mask] = np.interp(x[mask], x[~mask], data[~mask])
            
        return cleaned, mask
