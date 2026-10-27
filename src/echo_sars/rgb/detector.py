from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from ultralytics import YOLO

from echo_sars.fusion.models import DetectionBBox, ModalityEvidence


class RGBPersonDetector:
    """
    Lightweight person detector on RGB images using Ultralytics YOLOv8.
    Filters specifically for COCO class 0 ('person').
    """

    def __init__(self, model_weight: str = "yolov8n.pt", conf_thresh: float = 0.35):
        self.model_weight = model_weight
        self.conf_thresh = conf_thresh
        self.model = YOLO(model_weight)

    def detect(self, image_input: str | Path | np.ndarray) -> List[ModalityEvidence]:
        """
        Run inference on an RGB image (filepath or BGR numpy array).
        Returns a list of ModalityEvidence objects for each person detected.
        """
        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            results = self.model(image_path, conf=self.conf_thresh, verbose=False)
        else:
            results = self.model(image_input, conf=self.conf_thresh, verbose=False)

        evidences: List[ModalityEvidence] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                # COCO person class is 0
                if cls_id != 0:
                    continue

                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                bbox = DetectionBBox(
                    xmin=xyxy[0],
                    ymin=xyxy[1],
                    xmax=xyxy[2],
                    ymax=xyxy[3]
                )

                evidence = ModalityEvidence(
                    modality="rgb",
                    detected=True,
                    confidence=conf,
                    raw_details={
                        "class_id": cls_id,
                        "class_name": "person",
                        "model": self.model_weight,
                    },
                    bbox=bbox,
                    relative_position=None  # RGB 2D monocular does not provide direct relative 3D position
                )
                evidences.append(evidence)

        return evidences

    def draw_detections(self, image: np.ndarray, detections: List[ModalityEvidence]) -> np.ndarray:
        """Draw bounding boxes and confidence on an image."""
        annotated = image.copy()
        for det in detections:
            if det.bbox is None:
                continue
            p1 = (int(det.bbox.xmin), int(det.bbox.ymin))
            p2 = (int(det.bbox.xmax), int(det.bbox.ymax))
            cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
            label = f"Person: {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (p1[0], max(15, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        return annotated

