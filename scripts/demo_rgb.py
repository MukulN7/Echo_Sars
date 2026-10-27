import argparse
from pathlib import Path
import cv2

from echo_sars.config import FLIR_ROOT, YOLO_MODEL, CONF_THRESHOLD, OUTPUT_DIR
from echo_sars.rgb.dataset import FLIRDataset
from echo_sars.rgb.detector import RGBPersonDetector
from echo_sars.events.schema import SAREvent
from echo_sars.events.exporter import EventExporter
from echo_sars.fusion.engine import TransparentFusionEngine


def main():
    parser = argparse.ArgumentParser(description="ECHO-SARS RGB Person Detection Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to RGB image (optional, defaults to first FLIR val image)")
    parser.add_argument("--model", type=str, default=YOLO_MODEL, help="YOLO model checkpoint name or path")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated image and event")
    args = parser.parse_args()

    image_path = None
    if args.image:
        image_path = Path(args.image)
    else:
        val_rgb_dir = FLIR_ROOT / "images_rgb_val"
        if not val_rgb_dir.exists():
            raise FileNotFoundError(f"FLIR RGB validation directory not found at {val_rgb_dir}")
        dataset = FLIRDataset(val_rgb_dir)
        first_img = next(dataset.iter_images())
        image_path = dataset.get_image_path(first_img["id"])

    print(f"[*] Processing RGB Image: {image_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load detector
    detector = RGBPersonDetector(model_weight=args.model, conf_thresh=args.conf)
    raw_img = cv2.imread(str(image_path))
    if raw_img is None:
        raise ValueError(f"Could not decode image at {image_path}")

    # Run detection
    evidences = detector.detect(raw_img)
    print(f"[+] Detections found: {len(evidences)} person(s)")

    # Run fusion engine and format events
    fusion = TransparentFusionEngine()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, ev in enumerate(evidences):
        hypothesis = fusion.fuse(rgb_evidence=ev)
        event = SAREvent.from_hypothesis(hypothesis, notes=f"Source: {image_path.name}")
        print(f"\n--- Candidate #{i+1} ---")
        print(f"Confidence: {ev.confidence:.3f}")
        print(f"Bounding Box: [{ev.bbox.xmin:.1f}, {ev.bbox.ymin:.1f}, {ev.bbox.xmax:.1f}, {ev.bbox.ymax:.1f}]")
        print(f"Relative Position: {event.relative_position}")
        print(f"Operational Status: {event.operational_status}")

        if args.save:
            event_out = OUTPUT_DIR / f"demo_rgb_event_{i+1}.json"
            EventExporter.save_event(event, event_out)
            print(f"Exported SAR Event to {event_out}")

    if args.save:
        annotated = detector.draw_detections(raw_img, evidences)
        out_img_path = OUTPUT_DIR / "demo_rgb_detected.jpg"
        cv2.imwrite(str(out_img_path), annotated)
        print(f"\n[+] Saved annotated detection visualization to {out_img_path}")


if __name__ == "__main__":
    main()

