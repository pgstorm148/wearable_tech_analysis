import argparse
from biometric_wearable.pipeline import BiometricPipeline

def main():
    parser = argparse.ArgumentParser(description="Biometric Wearable Simulation System")
    parser.add_argument("--duration", type=int, default=60, help="Session duration in seconds")
    parser.add_argument("--hr", type=float, default=65.0, help="Base heart rate BPM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-dashboard", action="store_true", help="Run headless (skip matplotlib)")
    parser.add_argument("--output-dir", type=str, default="./output", help="Where to save logs")
    parser.add_argument("--stress-at", type=str, default="", help="Comma-separated seconds to inject stress events")
    parser.add_argument("--nfc-taps", type=int, default=5, help="Number of random NFC events to simulate")
    
    args = parser.parse_args()
    
    stress_at = [int(x.strip()) for x in args.stress_at.split(",")] if args.stress_at else []
    
    pipeline = BiometricPipeline(
        duration=args.duration,
        hr=args.hr,
        seed=args.seed,
        output_dir=args.output_dir,
        stress_at=stress_at,
        nfc_taps=args.nfc_taps,
        use_dashboard=not args.no_dashboard
    )
    
    pipeline.run()

if __name__ == "__main__":
    main()
