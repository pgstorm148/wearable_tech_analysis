import numpy as np
from typing import Tuple, Dict, List, Callable

# Constants
EMG_SAMPLE_RATE = 1000
HRV_SAMPLE_RATE = 250
IMU_SAMPLE_RATE = 100

class EMGGenerator:
    """Generates synthetic EMG signals with sub-vocal bursts and motion artifacts."""
    def __init__(self, sample_rate: int = EMG_SAMPLE_RATE, duration: int = 60, seed: int = 42):
        self.fs = sample_rate
        self.duration = duration
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, self.duration, self.duration * self.fs, endpoint=False)
        
        # Pink noise approximation
        white_noise = self.rng.standard_normal(len(t))
        pink_noise = np.cumsum(white_noise)
        pink_noise = pink_noise - np.mean(pink_noise)
        pink_noise = pink_noise / np.std(pink_noise) * 20.0 # 20 uV background
        
        signal = pink_noise.copy()
        
        # Add bursts (sub-vocal)
        num_bursts = self.duration * 2 # 2 bursts per second
        burst_times = self.rng.uniform(0, self.duration, num_bursts)
        for bt in burst_times:
            burst_duration = self.rng.uniform(0.05, 0.2)
            burst_samples = int(burst_duration * self.fs)
            start_idx = int(bt * self.fs)
            if start_idx + burst_samples < len(t):
                window = np.hanning(burst_samples)
                burst = self.rng.standard_normal(burst_samples) * self.rng.uniform(150, 400)
                signal[start_idx:start_idx+burst_samples] += burst * window
                
        # Add motion artifacts every 3-5s
        t_artifact = 0.0
        while t_artifact < self.duration:
            t_artifact += self.rng.uniform(3.0, 5.0)
            if t_artifact < self.duration:
                art_samples = int(0.5 * self.fs)
                start_idx = int(t_artifact * self.fs)
                if start_idx + art_samples < len(t):
                    window = np.hanning(art_samples)
                    artifact = self.rng.standard_normal(art_samples) * 1000.0 # large spike
                    signal[start_idx:start_idx+art_samples] += artifact * window
                    
        return t, signal

class HRVGenerator:
    """Generates synthetic PPG/ECG-like signals and HRV metrics."""
    def __init__(self, sample_rate: int = HRV_SAMPLE_RATE, duration: int = 60, base_hr: float = 65.0, stress_at: List[int] = None, seed: int = 42):
        self.fs = sample_rate
        self.duration = duration
        self.base_hr = base_hr
        self.stress_at = stress_at or []
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def generate(self) -> Tuple[np.ndarray, np.ndarray, Callable[[float], Dict]]:
        t = np.linspace(0, self.duration, self.duration * self.fs, endpoint=False)
        signal = np.zeros_like(t)
        
        current_time = 0.0
        rr_intervals = []
        peak_times = []
        
        while current_time < self.duration:
            is_stress = any(s <= current_time <= s + 10 for s in self.stress_at)
            current_hr = self.base_hr + 30.0 if is_stress else self.base_hr
            
            hrv_std = 0.02 if is_stress else 0.05
            rr = (60.0 / current_hr) + self.rng.normal(0, hrv_std)
            rr = max(0.3, rr)
            
            rr_intervals.append(rr)
            current_time += rr
            if current_time < self.duration:
                peak_times.append(current_time)
                
        for pt in peak_times:
            idx = int(pt * self.fs)
            if idx < len(t):
                signal[idx] = 1.0
                
        kernel_t = np.linspace(-0.2, 0.4, int(0.6 * self.fs))
        p_wave = 0.1 * np.exp(-((kernel_t + 0.15) ** 2) / (2 * 0.02**2))
        qrs = 1.0 * np.exp(-(kernel_t ** 2) / (2 * 0.01**2)) - 0.2 * np.exp(-((kernel_t - 0.02) ** 2) / (2 * 0.01**2)) - 0.2 * np.exp(-((kernel_t + 0.02) ** 2) / (2 * 0.01**2))
        t_wave = 0.2 * np.exp(-((kernel_t - 0.25) ** 2) / (2 * 0.04**2))
        
        kernel = p_wave + qrs + t_wave
        signal = np.convolve(signal, kernel, mode='same')
        
        baseline = 0.5 * np.sin(2 * np.pi * 0.2 * t)
        noise = self.rng.normal(0, 0.02, len(t))
        signal += baseline + noise
        
        def get_metrics(timestamp_ms: float) -> Dict:
            t_sec = timestamp_ms / 1000.0
            is_stress = any(s <= t_sec <= s + 10 for s in self.stress_at)
            current_hr = self.base_hr + 30.0 if is_stress else self.base_hr
            rmssd = 20.0 if is_stress else 50.0 + self.rng.normal(0, 2.0)
            return {"mean_hr": current_hr, "rmssd_ms": rmssd}
            
        return t, signal, get_metrics

class IMUGenerator:
    """Generates synthetic 6-axis IMU data (accelerometer + gyroscope)."""
    def __init__(self, sample_rate: int = IMU_SAMPLE_RATE, duration: int = 60, seed: int = 42):
        self.fs = sample_rate
        self.duration = duration
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, self.duration, self.duration * self.fs, endpoint=False)
        
        accel = np.zeros((len(t), 3))
        accel[:, 2] = 9.81 # Gravity
        
        num_movements = self.duration // 5
        move_times = self.rng.uniform(0, self.duration, num_movements)
        for mt in move_times:
            move_dur = self.rng.uniform(1.0, 3.0)
            start_idx = int(mt * self.fs)
            samples = int(move_dur * self.fs)
            if start_idx + samples < len(t):
                window = np.hanning(samples)[:, np.newaxis]
                movement = self.rng.normal(0, 2.0, (samples, 3))
                accel[start_idx:start_idx+samples] += movement * window
                
        gyro = np.zeros((len(t), 3))
        drift = np.cumsum(self.rng.normal(0, 0.01, (len(t), 3)), axis=0)
        gyro += drift
        
        for mt in move_times:
            move_dur = self.rng.uniform(1.0, 3.0)
            start_idx = int(mt * self.fs)
            samples = int(move_dur * self.fs)
            if start_idx + samples < len(t):
                window = np.hanning(samples)[:, np.newaxis]
                movement = self.rng.normal(0, 1.0, (samples, 3))
                gyro[start_idx:start_idx+samples] += movement * window
                
        accel += self.rng.normal(0, 0.1, accel.shape)
        gyro += self.rng.normal(0, 0.05, gyro.shape)
        
        imu_data = np.hstack((accel, gyro))
        return t, imu_data
