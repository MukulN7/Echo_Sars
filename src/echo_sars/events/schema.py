import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from echo_sars.fusion.models import RelativePosition, ModalityEvidence, FusedHypothesis


class SAREvent(BaseModel):
    """
    Structured SAR Detection Event schema.
    Exported for drone telemetry, base station logs, or mission planning.
    Ensures relative_position is strictly null when unknown.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_type: str = "PERSON_SEARCH_AND_RESCUE_CANDIDATE"
    target_class: str = "person"
    fused_score: float = Field(..., ge=0.0, le=1.0)
    relative_position: Optional[RelativePosition] = None
    modalities_used: List[str] = Field(default_factory=list)
    evidence: Dict[str, ModalityEvidence] = Field(default_factory=dict)
    operational_status: str = Field(
        "UNVERIFIED",
        description="UNVERIFIED, CANDIDATE_CONFIRMED, LOW_CONFIDENCE, AMBIGUOUS"
    )
    notes: str = ""

    @classmethod
    def from_hypothesis(
        cls,
        hypothesis: FusedHypothesis,
        override_timestamp: Optional[str] = None,
        notes: str = ""
    ) -> "SAREvent":
        # Categorize operational status
        score = hypothesis.fused_score
        if score >= 0.75 and ("rgb" in hypothesis.modalities_present or "lidar" in hypothesis.modalities_present):
            status = "CANDIDATE_CONFIRMED"
        elif score >= 0.40:
            status = "UNVERIFIED"
        else:
            status = "LOW_CONFIDENCE"

        event_ts = override_timestamp or datetime.now(timezone.utc).isoformat()
        combined_notes = hypothesis.fusion_notes
        if notes:
            combined_notes = f"{combined_notes}; {notes}".strip("; ")

        return cls(
            timestamp=event_ts,
            fused_score=score,
            relative_position=hypothesis.relative_position,
            modalities_used=hypothesis.modalities_present,
            evidence=hypothesis.evidence,
            operational_status=status,
            notes=combined_notes
        )

