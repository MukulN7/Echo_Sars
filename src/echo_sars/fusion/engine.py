from typing import Dict, List, Optional
from echo_sars.fusion.models import ModalityEvidence, FusedHypothesis, RelativePosition


class TransparentFusionEngine:
    """
    Transparent, modular evidence combination engine for SAR person detection.

    Design Principles:
      1. Preserves each sensor's original raw evidence untouched.
      2. Missing sensors are handled explicitly without hallucinating missing data.
      3. Heat alone (thermal contrast) is NEVER treated as proof of a person.
         Thermal alone can at most yield low/speculative interest.
      4. LiDAR provides accurate metric relative position (range/bearing), while RGB
         provides visual confirmation of human anatomy / clothing.
      5. Explicit explanations are attached in fusion_notes.
    """

    def __init__(
        self,
        rgb_weight: float = 0.50,
        thermal_weight: float = 0.20,
        lidar_weight: float = 0.30,
        min_report_threshold: float = 0.30
    ):
        self.weights = {
            "rgb": rgb_weight,
            "thermal": thermal_weight,
            "lidar": lidar_weight,
        }
        self.min_report_threshold = min_report_threshold

    def fuse(
        self,
        rgb_evidence: Optional[ModalityEvidence] = None,
        thermal_evidence: Optional[ModalityEvidence] = None,
        lidar_evidence: Optional[ModalityEvidence] = None,
    ) -> FusedHypothesis:
        evidence_map: Dict[str, ModalityEvidence] = {}
        present_modalities: List[str] = []

        if rgb_evidence is not None:
            evidence_map["rgb"] = rgb_evidence
            present_modalities.append("rgb")
        if thermal_evidence is not None:
            evidence_map["thermal"] = thermal_evidence
            present_modalities.append("thermal")
        if lidar_evidence is not None:
            evidence_map["lidar"] = lidar_evidence
            present_modalities.append("lidar")

        if not present_modalities:
            return FusedHypothesis(
                target_class="person",
                fused_score=0.0,
                modalities_present=[],
                evidence={},
                relative_position=None,
                fusion_notes="No sensor modalities provided."
            )

        # Evaluate individual contributions
        total_weight = 0.0
        weighted_sum = 0.0
        notes = []

        # Check for thermal-only scenario (Strict safeguard: heat alone != person)
        if set(present_modalities) == {"thermal"}:
            t_conf = thermal_evidence.confidence if thermal_evidence else 0.0
            capped_score = min(0.35, t_conf * 0.5)
            return FusedHypothesis(
                target_class="person",
                fused_score=round(capped_score, 3),
                modalities_present=["thermal"],
                evidence=evidence_map,
                relative_position=None,
                fusion_notes=(
                    "Thermal contrast alone detected; unconfirmed by RGB shape or LiDAR geometry. "
                    "Cannot confirm person based on heat signature alone."
                )
            )

        for mod in present_modalities:
            ev = evidence_map[mod]
            w = self.weights.get(mod, 0.2)
            total_weight += w
            if ev.detected:
                weighted_sum += w * ev.confidence
                notes.append(f"{mod.upper()} positive (conf={ev.confidence:.2f})")
            else:
                notes.append(f"{mod.upper()} negative/no detection")

        normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Bonus synergy: RGB person confirmed by LiDAR distance geometry
        has_rgb_det = rgb_evidence and rgb_evidence.detected and rgb_evidence.confidence > 0.4
        has_lidar_det = lidar_evidence and lidar_evidence.detected and lidar_evidence.confidence > 0.4
        has_thermal_det = thermal_evidence and thermal_evidence.detected and thermal_evidence.confidence > 0.4

        if has_rgb_det and has_lidar_det:
            normalized_score = min(0.99, normalized_score + 0.15)
            notes.append("Multi-modal synergy: RGB visual shape corroborated by LiDAR metric position.")

        if has_thermal_det and (has_rgb_det or has_lidar_det):
            normalized_score = min(0.99, normalized_score + 0.05)
            notes.append("Thermal contrast supports physical detection.")

        # Resolve relative 3D/2D position:
        # Prioritize LiDAR metric position if available; else null
        fused_pos: Optional[RelativePosition] = None
        if lidar_evidence and lidar_evidence.relative_position is not None:
            fused_pos = lidar_evidence.relative_position
        elif rgb_evidence and rgb_evidence.relative_position is not None:
            fused_pos = rgb_evidence.relative_position
        else:
            fused_pos = None  # Explicitly remain null when unknown

        return FusedHypothesis(
            target_class="person",
            fused_score=round(normalized_score, 3),
            modalities_present=present_modalities,
            evidence=evidence_map,
            relative_position=fused_pos,
            fusion_notes="; ".join(notes)
        )

