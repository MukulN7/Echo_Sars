from pathlib import Path
import pytest
from echo_sars.lidar.frog_loader import FROGDataset
from echo_sars.lidar.detector import LiDARBaselineDetector
from echo_sars.config import FROG_TEST


def test_frog_dataset_loading():
    if not FROG_TEST.exists():
        pytest.skip(f"FROG test dataset not found: {FROG_TEST}")

    with FROGDataset(FROG_TEST) as dataset:
        assert len(dataset) > 0
        scan = dataset.get_scan(0)
        assert "ranges" in scan
        assert "timestamp" in scan
        assert "num_persons" in scan
        assert "persons" in scan
        assert scan["ranges"].shape == (720,)


def test_frog_scan_to_cartesian():
    if not FROG_TEST.exists():
        pytest.skip(f"FROG test dataset not found: {FROG_TEST}")

    with FROGDataset(FROG_TEST) as dataset:
        scan = dataset.get_scan(0)
        x, y = FROGDataset.scan_to_cartesian(scan["ranges"])
        assert len(x) == len(y)
        assert len(x) > 0


def test_frog_detector_baseline():
    if not FROG_TEST.exists():
        pytest.skip(f"FROG test dataset not found: {FROG_TEST}")

    with FROGDataset(FROG_TEST) as dataset:
        scan = dataset.get_scan(0)
        detector = LiDARBaselineDetector()
        evidences = detector.detect(scan)
        assert isinstance(evidences, list)
        if len(evidences) > 0:
            ev = evidences[0]
            assert ev.modality == "lidar"
            assert ev.relative_position is not None
            assert ev.relative_position.range_m is not None

