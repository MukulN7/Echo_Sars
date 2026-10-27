import argparse
from pathlib import Path

from echo_sars.config import FROG_TEST, OUTPUT_DIR
from echo_sars.lidar.frog_loader import FROGDataset
from echo_sars.lidar.detector import LiDARBaselineDetector
from echo_sars.lidar.visualizer import plot_frog_scan
from echo_sars.fusion.engine import TransparentFusionEngine
from echo_sars.events.schema import SAREvent
from echo_sars.events.exporter import EventExporter


def main():
    parser = argparse.ArgumentParser(description="ECHO-SARS LiDAR FROG Scan Demo")
    parser.add_argument("--h5", type=str, default=str(FROG_TEST), help="Path to FROG HDF5 file")
    parser.add_argument("--index", type=int, default=0, help="Scan index to load and visualize")
    parser.add_argument("--save", action="store_true", default=True, help="Save scan visualization and event")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    print(f"[*] Opening FROG LiDAR dataset: {h5_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"FROG dataset missing at {h5_path}")

    dataset = FROGDataset(h5_path)
    print(f"[+] Total scans available: {len(dataset)}")

    scan_data = dataset.get_scan(args.index)
    print(f"[+] Loaded Scan #{args.index} (Timestamp: {scan_data['timestamp']}, Annotated Persons: {scan_data['num_persons']})")

    # Run LiDAR Baseline Detector
    detector = LiDARBaselineDetector()
    evidences = detector.detect(scan_data)
    print(f"[+] Extracted {len(evidences)} LiDAR person detection evidence(s)")

    fusion = TransparentFusionEngine()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, ev in enumerate(evidences):
        hypothesis = fusion.fuse(lidar_evidence=ev)
        event = SAREvent.from_hypothesis(hypothesis, notes=f"FROG Scan #{args.index}")
        print(f"\n--- LiDAR Target #{i+1} ---")
        print(f"Confidence: {ev.confidence:.3f}")
        print(f"Relative Position: x={ev.relative_position.x:.2f}m, y={ev.relative_position.y:.2f}m, range={ev.relative_position.range_m:.2f}m, azimuth={ev.relative_position.azimuth_deg:.1f}deg")
        print(f"Fused Score: {event.fused_score:.3f}")
        print(f"Status: {event.operational_status}")

        if args.save:
            event_out = OUTPUT_DIR / f"demo_lidar_event_{i+1}.json"
            EventExporter.save_event(event, event_out)
            print(f"Exported event to {event_out}")

    if args.save:
        out_plot_path = OUTPUT_DIR / f"demo_lidar_scan_{args.index}.png"
        plot_frog_scan(scan_data, save_path=out_plot_path, show=False)
        print(f"\n[+] Saved scan visualization plot to {out_plot_path}")

    dataset.close()


if __name__ == "__main__":
    main()

