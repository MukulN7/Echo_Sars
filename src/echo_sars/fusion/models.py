from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DetectionBBox(BaseModel):
    """2D Bounding box in pixels: [xmin, ymin, xmax, ymax]."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height


class RelativePosition(BaseModel):
    """
    Relative position of target in sensor / drone coordinates (in meters).
    If unknown or indeterminate, fields remain None.
    """
    x: Optional[float] = Field(None, description="Forward distance (+x in meters) or relative lateral offset")
    y: Optional[float] = Field(None, description="Lateral distance (+y in meters)")
    z: Optional[float] = Field(None, description="Vertical distance (+z in meters)")
    range_m: Optional[float] = Field(None, description="Radial distance in meters")
    azimuth_deg: Optional[float] = Field(None, description="Azimuth angle in degrees")


class ModalityEvidence(BaseModel):
    """Individual modality observation and confidence."""
    modality: str  # 'rgb', 'thermal', 'lidar'
    detected: bool = False
    confidence: float = 0.0
    raw_details: Dict[str, Any] = Field(default_factory=dict)
    bbox: Optional[DetectionBBox] = None
    relative_position: Optional[RelativePosition] = None


class FusedHypothesis(BaseModel):
    """Fused target hypothesis preserving each modality's evidence without false certainty."""
    target_class: str = "person"
    fused_score: float = 0.0
    modalities_present: List[str] = Field(default_factory=list)
    evidence: Dict[str, ModalityEvidence] = Field(default_factory=dict)
    relative_position: Optional[RelativePosition] = None
    fusion_notes: str = ""

