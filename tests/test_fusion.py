import pytest
from echo_sars.fusion.engine import TransparentFusionEngine
from echo_sars.fusion.models import ModalityEvidence, DetectionBBox, RelativePosition


def test_fusion_no_modalities():
    engine = TransparentFusionEngine()
    result = engine.fuse()
    assert result.fused_score == 0.0
    assert result.modalities_present == []
    assert result.relative_position is None


def test_heat_alone_is_not_proof_of_person():
    """Thermal contrast alone must NEVER produce high certainty of a person."""
    engine = TransparentFusionEngine()
    thermal_ev = ModalityEvidence(
        modality="thermal",
        detected=True,
        confidence=0.90,  # High thermal intensity / contrast
        raw_details={"is_calibrated_temperature": False}
    )
    result = engine.fuse(thermal_evidence=thermal_ev)

    # Score must be conservative and capped, not > 0.35
    assert result.fused_score <= 0.35
    assert "Thermal contrast alone" in result.fusion_notes


def test_missing_sensor_graceful_handling():
    engine = TransparentFusionEngine()
    # Only RGB present
    rgb_ev = ModalityEvidence(
        modality="rgb",
        detected=True,
        confidence=0.80,
        bbox=DetectionBBox(xmin=10, ymin=10, xmax=50, ymax=100)
    )
    result = engine.fuse(rgb_evidence=rgb_ev)
    assert "rgb" in result.modalities_present
    assert "lidar" not in result.modalities_present
    assert result.fused_score > 0.0
    # RGB alone has no 3D relative positioning, so relative_position must be null
    assert result.relative_position is None


def test_multimodal_synergy():
    engine = TransparentFusionEngine()
    rgb_ev = ModalityEvidence(
        modality="rgb",
        detected=True,
        confidence=0.85,
        bbox=DetectionBBox(xmin=10, ymin=10, xmax=50, ymax=100)
    )
    lidar_pos = RelativePosition(x=5.2, y=-1.1, range_m=5.32, azimuth_deg=-11.9)
    lidar_ev = ModalityEvidence(
        modality="lidar",
        detected=True,
        confidence=0.88,
        relative_position=lidar_pos
    )

    result = engine.fuse(rgb_evidence=rgb_ev, lidar_evidence=lidar_ev)
    assert result.fused_score > 0.85
    assert result.relative_position == lidar_pos
    assert "Multi-modal synergy" in result.fusion_notes

