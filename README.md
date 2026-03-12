# Biometric Wearable Simulation System

A complete Python simulation of a real-time physiological data acquisition pipeline with NFC integration.

## Architecture
```text
[Sensors] -> [Signal Processing] -> [Sensor Fusion] -> [Dashboard & Logs]
  |                 |                      |
 EMG (1000Hz) -> Bandpass/Notch -> RMS/Freq Drop \
 HRV (250Hz)  -> Peak Detect    -> RMSSD/BPM      -> Fatigue Index, Motion State
 IMU (100Hz)  -> Kalman Filter  -> Accel Mag     /
  |
 NFC (Touch-to-Action) -> State Control (Start/Pause/Sync)
```

## Installation & Usage
```bash
pip install numpy scipy matplotlib pandas
python run.py --duration 30 --stress-at 10,20 --nfc-taps 4
```

## Key Design Decisions
- **Kalman filter vs simple averaging**: Uses covariance tracking for optimal noise reduction on IMU data, simulating real wearable edge processing.
- **FFT bandpass for EMG**: Isolates the 20-450Hz muscle activation band and removes 50Hz power line noise.
- **NFC NDEF payload**: Implements touch-to-action logic (start/pause/end) for low-power authentication without BLE overhead.
- **Sensor fusion fatigue model**: Combines normalized HRV (RMSSD) and EMG median frequency drop to estimate real-time muscle fatigue.

## Output
- **Dashboard**: Real-time matplotlib visualization of EMG, HRV, IMU, and Fatigue Index.
- **Logs**: Saves `fusion_log.csv` (timestamped metrics) and `nfc_events.json` to the `--output-dir`.

## Interview Talking Points
- **Edge inference**: Designed to simulate on-device processing with <50ms latency per tick.
- **Kalman filter**: A simplified 1D version of what runs on real wearables for motion artifact removal.
- **NFC**: Acts as a low-power authentication layer.
- **HRV**: Used as the ground truth for athlete recovery state and stress response.
