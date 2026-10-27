from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import h5py
import numpy as np


class FROGDataset:
    """
    Loader for the FROG (Forest Range and Optical Ground/Grid) 2D LiDAR dataset.

    Array layout:
      - scans: (N, 720) float32, range measurements in meters (0 to 360 deg or FOV, inf for no-return).
      - timestamps: (N,) float64.
      - circle_num: (N,) int/uint32, number of annotated person circle instances in each scan.
      - circle_idx: (N,) int/uint32, starting index into `circles` array.
      - circles: (M, 6) float32, target parameters:
          [x_cartesian, y_cartesian, radius, range_polar, azimuth_rad, radius_polar]
      - split: (N,) uint8, optional split indicator (0=train, 1=val).
    """

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"FROG HDF5 dataset not found at {self.h5_path}")
        self._h5_file: Optional[h5py.File] = None
        self._open()

    def _open(self) -> None:
        self._h5_file = h5py.File(str(self.h5_path), "r")

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self) -> int:
        if self._h5_file is None:
            return 0
        return len(self._h5_file["scans"])

    def get_scan(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single scan and its ground-truth person annotations.
        """
        if self._h5_file is None:
            self._open()

        f = self._h5_file
        scan_ranges = np.array(f["scans"][idx], dtype=np.float32)
        timestamp = float(f["timestamps"][idx])
        num_circles = int(f["circle_num"][idx])
        start_idx = int(f["circle_idx"][idx])

        person_circles = []
        if num_circles > 0:
            circles_slice = f["circles"][start_idx : start_idx + num_circles]
            for c in circles_slice:
                person_circles.append({
                    "x_m": float(c[0]),
                    "y_m": float(c[1]),
                    "radius_m": float(c[2]),
                    "range_m": float(c[3]),
                    "azimuth_rad": float(c[4]),
                    "azimuth_deg": float(np.rad2deg(c[4])),
                })

        return {
            "index": idx,
            "timestamp": timestamp,
            "ranges": scan_ranges,
            "num_persons": num_circles,
            "persons": person_circles,
        }

    @staticmethod
    def scan_to_cartesian(
        ranges: np.ndarray,
        fov_deg: float = 360.0,
        max_range: float = 30.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert range array into (x, y) cartesian coordinates.
        Filters out inf / out-of-range points.
        """
        n_rays = len(ranges)
        # Uniform angle distribution around scanner
        angles = np.linspace(0, np.deg2rad(fov_deg), n_rays, endpoint=False)
        valid = np.isfinite(ranges) & (ranges > 0.05) & (ranges < max_range)

        r = ranges[valid]
        theta = angles[valid]

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

