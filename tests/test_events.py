import json
import pytest
from echo_sars.events.schema import SAREvent
from echo_sars.events.exporter import EventExporter
from echo_sars.fusion.models import FusedHypothesis, ModalityEvidence, RelativePosition


def test_sar_event_export_and_null_position():
    hyp = FusedHypothesis(
        target_class="person",
        fused_score=0.82,
        modalities_present=["rgb"],
        evidence={"rgb": ModalityEvidence(modality="rgb", detected=True, confidence=0.82)},
        relative_position=None,  # Unknown relative position
        fusion_notes="Visual detection without spatial depth"
    )
    event = SAREvent.from_hypothesis(hyp)
    assert event.operational_status == "CANDIDATE_CONFIRMED"
    assert event.relative_position is None

    # Serialization test
    json_str = EventExporter.to_json(event)
    data = json.loads(json_str)
    assert data["relative_position"] is None
    assert data["target_class"] == "person"
    assert data["fused_score"] == 0.82


def test_sar_event_with_lidar_position(tmp_path):
    hyp = FusedHypothesis(
        target_class="person",
        fused_score=0.89,
        modalities_present=["rgb", "lidar"],
        relative_position=RelativePosition(x=10.2, y=-1.5, range_m=10.31, azimuth_deg=-8.3),
        fusion_notes="Corroborated detection"
    )
    event = SAREvent.from_hypothesis(hyp)
    event_file = tmp_path / "event.json"
    EventExporter.save_event(event, event_file)

    assert event_file.exists()
    with open(event_file, "r") as f:
        loaded = json.load(f)
    assert loaded["relative_position"]["x"] == 10.2
    assert loaded["relative_position"]["range_m"] == 10.31

