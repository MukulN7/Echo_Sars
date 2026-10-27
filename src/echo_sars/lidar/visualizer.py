from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

from echo_sars.lidar.frog_loader import FROGDataset


def plot_frog_scan(
    scan_data: Dict[str, Any],
    save_path: Optional[str | Path] = None,
    show: bool = False
) -> None:
    """
    Visualize a 2D LiDAR scan and overlaid ground-truth person circles.
    """
    ranges = scan_data["ranges"]
    x, y = FROGDataset.scan_to_cartesian(ranges)

    fig, ax = plt.subplots(figsize=(8, 8))
    # Scatter point cloud
    ax.scatter(x, y, s=4, c="navy", alpha=0.6, label="LiDAR points")
    ax.scatter([0], [0], s=80, c="red", marker="x", label="Scanner / Drone origin")

    # Annotate person circles
    persons = scan_data.get("persons", [])
    for idx, p in enumerate(persons):
        circle = plt.Circle(
            (p["x_m"], p["y_m"]),
            p["radius_m"],
            color="crimson",
            fill=False,
            linewidth=2,
            linestyle="--",
            label="Person annotation" if idx == 0 else ""
        )
        ax.add_patch(circle)
        ax.text(
            p["x_m"] + 0.3,
            p["y_m"] + 0.3,
            f"P{idx+1} ({p['range_m']:.1f}m)",
            color="crimson",
            fontweight="bold",
            fontsize=9
        )

    ax.set_title(
        f"FROG LiDAR Scan #{scan_data['index']} (Persons: {len(persons)})",
        fontsize=12,
        pad=10
    )
    ax.set_xlabel("Lateral X (meters)")
    ax.set_ylabel("Forward Y (meters)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    ax.axis("equal")

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_p), dpi=150, bbox_inches="tight")
        print(f"Saved LiDAR plot to {save_p}")

    if show:
        plt.show()
    plt.close(fig)

