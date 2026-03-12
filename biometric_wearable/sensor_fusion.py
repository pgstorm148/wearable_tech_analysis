import numpy as np
from typing import Dict

class SensorFusionEngine:
    """Fuses EMG, HRV, and IMU data into a unified health state."""
    def __init__(self):
        self.baseline_emg_freq = 150.0 # Hz
        
    def process_frame(self, timestamp_ms: float, emg_filtered: np.ndarray, 
                      emg_dom_freq: float, hrv_metrics: Dict, 
                      imu_filtered: np.ndarray, latency_ms: float) -> Dict:
        
        emg_rms = np.sqrt(np.mean(emg_filtered**2)) if len(emg_filtered) > 0 else 0.0
        emg_activation = min(1.0, emg_rms / 200.0)
        
        if len(imu_filtered) > 0:
            accel_mag = np.sqrt(np.sum(imu_filtered[:, :3]**2, axis=1))
            mean_accel = np.mean(accel_mag)
        else:
            mean_accel = 9.81
            
        if mean_accel > 12.0:
            motion_state = "active"
        elif mean_accel > 10.5:
            motion_state = "gesture"
        else:
            motion_state = "rest"
            
        rmssd = hrv_metrics.get("rmssd_ms", 50.0)
        norm_hrv = min(1.0, max(0.0, rmssd / 100.0))
        
        freq_drop = max(0.0, (self.baseline_emg_freq - emg_dom_freq) / self.baseline_emg_freq)
        freq_drop = min(1.0, freq_drop)
        
        fatigue_index = 0.6 * (1.0 - norm_hrv) + 0.4 * freq_drop
        fatigue_index = min(1.0, max(0.0, fatigue_index))
        
        return {
            "timestamp_ms": timestamp_ms,
            "hrv_rmssd": rmssd,
            "hr_bpm": hrv_metrics.get("mean_hr", 65.0),
            "emg_activation": emg_activation,
            "motion_state": motion_state,
            "fatigue_index": fatigue_index,
            "latency_ms": latency_ms
        }
