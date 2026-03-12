import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque

class LiveDashboard:
    """Real-time matplotlib dashboard for biometric data."""
    def __init__(self, duration: int):
        self.duration = duration
        self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Biometric Wearable Dashboard")
        self.fig.tight_layout(pad=3.0)
        
        # Data queues
        self.emg_t = deque(maxlen=2000) # 2 seconds at 1000Hz
        self.emg_d = deque(maxlen=2000)
        
        self.hrv_t = deque(maxlen=500) # 2 seconds at 250Hz
        self.hrv_d = deque(maxlen=500)
        
        self.imu_t = deque(maxlen=300) # 3 seconds at 100Hz
        self.imu_accel = [deque(maxlen=300) for _ in range(3)]
        
        self.fatigue_t = deque(maxlen=duration * 10) # 10Hz fusion rate
        self.fatigue_d = deque(maxlen=duration * 10)
        
        self.nfc_events = []
        
        self.current_hr = 0
        self.current_rmssd = 0
        self.current_motion = "rest"
        self.current_time = 0
        self.emg_activation = 0
        
        # Setup plots
        self.line_emg, = self.axs[0].plot([], [], color='blue')
        self.axs[0].set_title("EMG Waveform (2s)")
        self.axs[0].set_ylim(-500, 500)
        self.axs[0].set_ylabel("µV")
        
        self.line_hrv, = self.axs[1].plot([], [], color='red')
        self.axs[1].set_title("PPG / HRV Waveform")
        self.axs[1].set_ylim(-2, 2)
        
        self.lines_imu = [self.axs[2].plot([], [], label=axis)[0] for axis in ['X', 'Y', 'Z']]
        self.axs[2].set_title("IMU Accelerometer (3s)")
        self.axs[2].set_ylim(5, 15)
        self.axs[2].legend(loc='upper right')
        
        self.line_fatigue, = self.axs[3].plot([], [], color='purple')
        self.axs[3].set_title("Fatigue Index")
        self.axs[3].set_ylim(0, 1)
        self.axs[3].set_xlim(0, duration)
        
    def update_data(self, emg_t, emg_d, hrv_t, hrv_d, imu_t, imu_d, fusion_data, nfc_events):
        self.emg_t.extend(emg_t)
        self.emg_d.extend(emg_d)
        
        self.hrv_t.extend(hrv_t)
        self.hrv_d.extend(hrv_d)
        
        self.imu_t.extend(imu_t)
        for i in range(3):
            self.imu_accel[i].extend(imu_d[:, i])
            
        self.fatigue_t.append(fusion_data["timestamp_ms"] / 1000.0)
        self.fatigue_d.append(fusion_data["fatigue_index"])
        
        self.current_hr = fusion_data["hr_bpm"]
        self.current_rmssd = fusion_data["hrv_rmssd"]
        self.current_motion = fusion_data["motion_state"]
        self.current_time = fusion_data["timestamp_ms"] / 1000.0
        self.emg_activation = fusion_data["emg_activation"]
        
        if nfc_events:
            self.nfc_events.extend(nfc_events)
            
    def _animate(self, frame):
        if not self.emg_t:
            return
            
        # Update EMG
        self.line_emg.set_data(self.emg_t, self.emg_d)
        self.axs[0].set_xlim(self.emg_t[0], self.emg_t[-1] + 0.001)
        
        # Update HRV
        self.line_hrv.set_data(self.hrv_t, self.hrv_d)
        self.axs[1].set_xlim(self.hrv_t[0], self.hrv_t[-1] + 0.001)
        self.axs[1].set_title(f"PPG / HRV Waveform | HR: {self.current_hr:.1f} BPM | RMSSD: {self.current_rmssd:.1f} ms")
        
        # Update IMU
        for i in range(3):
            self.lines_imu[i].set_data(self.imu_t, self.imu_accel[i])
        self.axs[2].set_xlim(self.imu_t[0], self.imu_t[-1] + 0.001)
        
        # Update Fatigue
        self.line_fatigue.set_data(self.fatigue_t, self.fatigue_d)
        
        # Color coding title based on fatigue
        fatigue = self.fatigue_d[-1] if self.fatigue_d else 0
        color = 'green' if fatigue < 0.4 else 'orange' if fatigue < 0.7 else 'red'
        
        self.fig.suptitle(f"Time: {self.current_time:.1f}s | Motion: {self.current_motion} | EMG Act: {self.emg_activation:.2f}", color=color, fontsize=14)
        
        # Draw NFC events
        for event in self.nfc_events:
            t_ev = event["timestamp_ms"] / 1000.0
            for ax in self.axs:
                has_line = any(hasattr(line, 'nfc_id') and line.nfc_id == event["event_id"] for line in ax.lines)
                if not has_line:
                    line = ax.axvline(x=t_ev, color='magenta', linestyle='--', alpha=0.7)
                    line.nfc_id = event["event_id"]
                    if ax == self.axs[0]:
                        ax.text(t_ev, ax.get_ylim()[1]*0.8, event["action"], color='magenta', rotation=90, verticalalignment='top')
                        
        return self.line_emg, self.line_hrv, self.line_fatigue
        
    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self._animate, interval=50, blit=False, cache_frame_data=False)
        plt.show()
