import time
import threading
import pandas as pd
import json
import os
import numpy as np
from biometric_wearable.synthetic_data import EMGGenerator, HRVGenerator, IMUGenerator
from biometric_wearable.signal_processor import KalmanFilter1D, FFTBandpassFilter, MotionArtifactRemover
from biometric_wearable.sensor_fusion import SensorFusionEngine
from biometric_wearable.nfc_simulator import NFCEventSimulator
from biometric_wearable.dashboard import LiveDashboard

class BiometricPipeline:
    """Orchestrates all modules in a main processing loop."""
    def __init__(self, duration: int, hr: float, seed: int, output_dir: str, stress_at: list, nfc_taps: int, use_dashboard: bool):
        self.duration = duration
        self.output_dir = output_dir
        self.use_dashboard = use_dashboard
        
        # Generators
        self.emg_gen = EMGGenerator(duration=duration, seed=seed)
        self.hrv_gen = HRVGenerator(duration=duration, base_hr=hr, stress_at=stress_at, seed=seed)
        self.imu_gen = IMUGenerator(duration=duration, seed=seed)
        
        # Processors
        self.emg_filter = FFTBandpassFilter(lowcut=20.0, highcut=450.0, fs=1000.0, notch_freq=50.0)
        self.emg_artifact_remover = MotionArtifactRemover()
        self.imu_kalman = [KalmanFilter1D() for _ in range(6)]
        
        self.fusion_engine = SensorFusionEngine()
        self.nfc_sim = NFCEventSimulator(duration=duration, num_taps=nfc_taps, seed=seed)
        
        if self.use_dashboard:
            self.dashboard = LiveDashboard(duration=duration)
            
        self.fusion_log = []
        self.nfc_log = []
        
    def run(self):
        print("Generating synthetic data...")
        emg_t, emg_sig = self.emg_gen.generate()
        hrv_t, hrv_sig, hrv_metrics_fn = self.hrv_gen.generate()
        imu_t, imu_sig = self.imu_gen.generate()
        
        print("Starting pipeline...")
        
        if self.use_dashboard:
            dash_thread = threading.Thread(target=self._run_pipeline, args=(emg_t, emg_sig, hrv_t, hrv_sig, hrv_metrics_fn, imu_t, imu_sig), daemon=True)
            dash_thread.start()
            self.dashboard.start() # This blocks until window is closed
        else:
            self._run_pipeline(emg_t, emg_sig, hrv_t, hrv_sig, hrv_metrics_fn, imu_t, imu_sig)
            
        self._save_logs()
        
    def _run_pipeline(self, emg_t, emg_sig, hrv_t, hrv_sig, hrv_metrics_fn, imu_t, imu_sig):
        tick_ms = 100
        current_ms = 0
        end_ms = self.duration * 1000
        
        emg_fs = 1000
        hrv_fs = 250
        imu_fs = 100
        
        try:
            while current_ms < end_ms:
                start_time = time.time()
                
                # Slicing indices
                emg_start, emg_end = int((current_ms/1000)*emg_fs), int(((current_ms+tick_ms)/1000)*emg_fs)
                hrv_start, hrv_end = int((current_ms/1000)*hrv_fs), int(((current_ms+tick_ms)/1000)*hrv_fs)
                imu_start, imu_end = int((current_ms/1000)*imu_fs), int(((current_ms+tick_ms)/1000)*imu_fs)
                
                if emg_end > len(emg_sig):
                    break
                    
                # 1. Pull window
                emg_w_t = emg_t[emg_start:emg_end]
                emg_w_d = emg_sig[emg_start:emg_end]
                
                hrv_w_t = hrv_t[hrv_start:hrv_end]
                hrv_w_d = hrv_sig[hrv_start:hrv_end]
                
                imu_w_t = imu_t[imu_start:imu_end]
                imu_w_d = imu_sig[imu_start:imu_end]
                
                # 2. Apply filters
                # EMG
                emg_clean, _ = self.emg_artifact_remover.process(emg_w_d)
                emg_filt, emg_dom_freq, _ = self.emg_filter.filter(emg_clean)
                
                # IMU Kalman
                imu_filt = np.zeros_like(imu_w_d)
                for i in range(len(imu_w_t)):
                    for axis in range(6):
                        imu_filt[i, axis] = self.imu_kalman[axis].update(imu_w_d[i, axis])
                        
                # 3. Fusion
                latency_ms = (time.time() - start_time) * 1000
                hrv_metrics = hrv_metrics_fn(current_ms)
                fusion_res = self.fusion_engine.process_frame(
                    timestamp_ms=current_ms,
                    emg_filtered=emg_filt,
                    emg_dom_freq=emg_dom_freq,
                    hrv_metrics=hrv_metrics,
                    imu_filtered=imu_filt,
                    latency_ms=latency_ms
                )
                self.fusion_log.append(fusion_res)
                
                # 4. NFC
                nfc_events = self.nfc_sim.get_events_in_window(current_ms, current_ms + tick_ms)
                if nfc_events:
                    self.nfc_log.extend(nfc_events)
                    for ev in nfc_events:
                        if ev["action"] == "end_session":
                            print("NFC end_session received. Stopping.")
                            return
                            
                # 5. Push to dashboard
                if self.use_dashboard:
                    self.dashboard.update_data(emg_w_t, emg_filt, hrv_w_t, hrv_w_d, imu_w_t, imu_filt, fusion_res, nfc_events)
                    
                current_ms += tick_ms
                
                # Sleep to simulate real-time
                elapsed = time.time() - start_time
                sleep_time = (tick_ms / 1000.0) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("Pipeline interrupted by user.")
            
    def _save_logs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.fusion_log:
            df = pd.DataFrame(self.fusion_log)
            csv_path = os.path.join(self.output_dir, "fusion_log.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved fusion log to {csv_path}")
            
        if self.nfc_log:
            json_path = os.path.join(self.output_dir, "nfc_events.json")
            with open(json_path, 'w') as f:
                json.dump(self.nfc_log, f, indent=2)
            print(f"Saved NFC events to {json_path}")
