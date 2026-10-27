from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from echo_sars.rgb.dataset import FLIRDataset
from echo_sars.fusion.models import DetectionBBox, ModalityEvidence


class ThermalInspector:
    """
    Thermal image inspector and evidence extractor for FLIR ADAS v2 thermal frames.

    CRITICAL RADIOMETRIC PRINCIPLE:
    In FLIR ADAS v2 (8-bit JPEG / uncalibrated raw values), pixel values represent
    relative thermal infrared radiance / optical sensor response, NOT calibrated
    absolute Celsius temperature. High pixel brightness can indicate thermal contrast
    or warm objects, but cannot be equated to body temperature or proof of life
    without sensor calibration metadata (NUC, emissivity, atmospheric transmission).
    """

    def __init__(self, dataset_split_dir: str | Path):
        self.dataset = FLIRDataset(dataset_split_dir)

    def load_frame(self, image_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load thermal frame and its metadata."""
        img_info = self.dataset.get_image_info(image_id)
        img_path = self.dataset.get_image_path(image_id)
        if not img_path.exists():
            raise FileNotFoundError(f"Thermal image file missing at {img_path}")

        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        return image, img_info

    def inspect_region(
        self,
        image: np.ndarray,
        bbox: DetectionBBox
    ) -> Dict[str, Any]:
        """
        Analyze pixel intensity distribution within a candidate bounding box.
        Emphasizes intensity vs. temperature distinction.
        """
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, int(bbox.xmin)))
        y1 = max(0, min(h - 1, int(bbox.ymin)))
        x2 = max(0, min(w, int(bbox.xmax)))
        y2 = max(0, min(h, int(bbox.ymax)))

        if x2 <= x1 or y2 <= y1:
            return {
                "valid": False,
                "error": "Degenerate bounding box coordinates"
            }

        crop = image[y1:y2, x1:x2]
        mean_val = float(np.mean(crop))
        max_val = float(np.max(crop))
        min_val = float(np.min(crop))
        std_val = float(np.std(crop))

        # Overall image background statistics for contrast comparison
        bg_mean = float(np.mean(image))
        contrast_ratio = (mean_val - bg_mean) / (bg_mean + 1e-5)

        return {
            "valid": True,
            "shape": crop.shape,
            "bit_depth": str(image.dtype),
            "intensity_mean": round(mean_val, 2),
            "intensity_max": round(max_val, 2),
            "intensity_min": round(min_val, 2),
            "intensity_std": round(std_val, 2),
            "background_mean": round(bg_mean, 2),
            "contrast_ratio": round(contrast_ratio, 3),
            "is_calibrated_temperature": False,
            "radiometric_note": (
                "Values represent uncalibrated 8-bit sensor intensity (0-255). "
                "Contrast does not equal verified body heat or human confirmation."
            )
        }

    def extract_thermal_evidence(
        self,
        image: np.ndarray,
        bbox: DetectionBBox
    ) -> ModalityEvidence:
        """
        Produce a conservative thermal ModalityEvidence from a region.
        Thermal contrast adds supportive weight, but does not claim certainty alone.
        """
        stats = self.inspect_region(image, bbox)
        if not stats.get("valid", False):
            return ModalityEvidence(
                modality="thermal",
                detected=False,
                confidence=0.0,
                raw_details={"error": stats.get("error")}
            )

        # Conservative heuristic: warm target against cooler background
        contrast = stats.get("contrast_ratio", 0.0)
        # Bounded thermal confidence [0.1 to 0.7 max] without shape/RGB confirmation
        thermal_confidence = min(0.70, max(0.10, 0.40 + contrast * 0.2))

        return ModalityEvidence(
            modality="thermal",
            detected=contrast > 0.05,
            confidence=round(thermal_confidence, 3),
            raw_details=stats,
            bbox=bbox,
            relative_position=None
        )

