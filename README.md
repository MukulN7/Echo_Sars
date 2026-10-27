# ECHO-SARS Intelligence v0.1

**Autonomous Search-and-Rescue (SAR) Drone Perception Prototype**

ECHO-SARS Intelligence v0.1 is a modular Python prototype for multi-modal person detection in search-and-rescue contexts using public datasets:
- **FLIR ADAS v2**: High-resolution RGB images and thermal infrared frames with COCO annotations.
- **FROG (Forest Range & Optical Ground)**: 2D LiDAR range scans with ground-truth target annotations in cluttered outdoor/forest environments.

---

## Architecture & Design Principles

```
                  +-----------------------------------+
                  |        Multi-Modal Inputs         |
                  +-----------------------------------+
                     /              |               \
        (FLIR RGB)  /        (FLIR Thermal)          \  (FROG LiDAR)
                   v                v                 v
            [RGB Detector]  [Thermal Inspector] [LiDAR Detector]
            (YOLOv8 Person) (Intensity != Temp) (Cluster / Metric)
                   \                |                 /
                    \               |                /
                     v              v               v
                  +-----------------------------------+
                  |     Transparent Fusion Engine     |
                  |  - Preserves original evidence    |
                  |  - Handles missing sensors        |
                  |  - Heat alone != proof of person  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |         Structured Event          |
                  |  - relative_position (m / null)   |
                  |  - operational_status & scores    |
                  |  - JSON / JSONL Telemetry schema  |
                  +-----------------------------------+
```

### Core Tenets:
1. **Explicit Separation of Datasets**: FLIR and FROG are collected by separate platforms and are not falsely claimed to be time-synchronized hardware streams.
2. **Thermal Radiometric Integrity**: In FLIR ADAS v2, pixel intensities represent optical infrared sensor radiance (8-bit JPEG 0-255), not calibrated absolute Celsius temperature. High contrast indicates warm thermal anomalies but **heat alone is never treated as proof of a person**.
3. **Metric Relative Localization**: When 2D LiDAR ranges and bearings are available, coordinates are computed in metric sensor coordinates (`x`, `y`, `range_m`, `azimuth_deg`). For monocular RGB or when depth is missing, `relative_position` **strictly remains `null`**.
4. **Resilient to Missing Sensors**: Operates seamlessly when only RGB, only LiDAR, or only Thermal is provided.

---

## Directory Structure

```
.
├── src/
│   └── echo_sars/
│       ├── __init__.py
│       ├── config.py             # Configurable paths via .env
│       ├── rgb/                  # FLIR COCO loader and YOLOv8 detector
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── detector.py
│       ├── thermal/              # Thermal contrast & radiometric inspector
│       │   ├── __init__.py
│       │   └── inspector.py
│       ├── lidar/                # FROG HDF5 scan loader, baseline & plot
│       │   ├── __init__.py
│       │   ├── frog_loader.py
│       │   ├── detector.py
│       │   └── visualizer.py
│       ├── fusion/               # Transparent evidence fusion
│       │   ├── __init__.py
│       │   ├── models.py
│       │   └── engine.py
│       └── events/               # Pydantic SAR event schema & JSON exporter
│           ├── __init__.py
│           ├── schema.py
│           └── exporter.py
├── scripts/
│   ├── demo_rgb.py               # Run RGB inference on single image
│   └── demo_lidar.py             # Load and visualize FROG LiDAR scan
├── tests/
│   ├── test_flir_loader.py       # FLIR COCO parsing tests
│   ├── test_frog_loader.py       # FROG HDF5 structure & range math
│   ├── test_fusion.py            # Missing inputs, heat safeguard & synergy
│   └── test_events.py            # JSON schema & relative position null checks
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation & Setup (Windows PowerShell)

Ensure Python 3.10+ (tested on Python 3.11) is installed.

### 1. Create and Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies & Local Package

```powershell
pip install -r requirements.txt
pip install -e .
```

### 3. Configure Dataset Paths

Copy `.env.example` to `.env` and adjust the paths to your local drive:

```powershell
Copy-Item .env.example .env
```

Default configuration in `.env`:
```env
ECHO_SARS_FLIR_ROOT=D:\Career\Competitons\Echo Sars\Dataset\FLIR_ADAS_v2
ECHO_SARS_FROG_TRAIN_VAL=D:\Career\Competitons\Echo Sars\Dataset\frog_11-36_12-43_train_val.h5
ECHO_SARS_FROG_TEST=D:\Career\Competitons\Echo Sars\Dataset\frog_16-41_test.h5
ECHO_SARS_YOLO_MODEL=yolov8n.pt
ECHO_SARS_CONF_THRESHOLD=0.35
ECHO_SARS_OUTPUT_DIR=outputs
```

---

## Running Demos

### 1. RGB Person Detection Demo

Runs lightweight YOLOv8 person detection on a FLIR validation frame, applies the fusion engine to construct hypotheses, and saves the annotated image and structured SAR event JSON to `outputs/`:

```powershell
python scripts/demo_rgb.py --image "Dataset/FLIR_ADAS_v2/images_rgb_val/data/video-CgGWcFRZQwyk48xKp-frame-002100-CRyvbujtPTuoZdLrK.jpg" --conf 0.25
```

**Artifacts Generated:**
- `outputs/demo_rgb_detected.jpg` (Bounding box overlay)
- `outputs/demo_rgb_event_1.json` (Structured JSON event with `relative_position: null`)

### 2. FROG LiDAR Scan & Detection Demo

Loads 2D range scans from the FROG HDF5 file (720 beams per scan), runs baseline detection, extracts metric relative coordinates, and generates a polar/cartesian scatter plot:

```powershell
python scripts/demo_lidar.py --index 0
```

**Artifacts Generated:**
- `outputs/demo_lidar_scan_0.png` (2D point cloud with target marker)
- `outputs/demo_lidar_event_1.json` (Structured JSON event containing metric relative coordinates `x`, `y`, `range_m`, `azimuth_deg`)

---

## Running the Test Suite

Run unit and integration tests across dataset parsing, fusion safeguards, and event schemas:

```powershell
pytest -v
```

All 11 tests verify:
- FLIR COCO JSON parsing and image resolution.
- FROG HDF5 key retrieval, ray-to-cartesian geometry conversion.
- Fusion engine behavior under missing sensors and multi-modal synergy.
- Strict thermal safeguard (heat alone capped at ≤ 0.35 confidence).
- Event serialization verifying `relative_position` stays `null` unless physical metric depth is present.

---

## Verified Capabilities & Incomplete Roadmap

### What Actually Works:
- Verified loaders for FLIR ADAS v2 (RGB & Thermal COCO annotations).
- Verified HDF5 scan readers for FROG dataset (`scans`, `circles`, `circle_num`, `timestamps`).
- Ultralytics YOLOv8 inference filtering for person instances.
- Thermal contrast extraction with clear radiometric warnings regarding calibrated temperatures.
- Baseline 2D LiDAR range clustering and metric relative coordinate extraction.
- Transparent fusion engine with missing-sensor resilience and thermal safeguards.
- Standardized Pydantic SAR event schema with strict JSON serialization.
- Automated tests covering dataset parsing, fusion logic, and event formatting.

### What Remains Incomplete / Future Work:
- **Spatial Multi-Modal Calibration**: FLIR and FROG datasets are currently separate domains. True cross-modal sensor extrinsic calibration (RGB-Thermal homography & LiDAR-to-camera matrix) requires a physical synchronized test rig.
- **Elevation / 3D Bounding Boxes**: FROG is planar 2D LiDAR; full 3D point cloud object estimation (e.g. 3D LiDAR point pillars) is reserved for future sensor integrations.
- **Edge Deployment & Hardware Loop**: Flight control integration (PX4 / MAVLink / ROS 2) is deliberately omitted in this v0.1 prototype until flight hardware is selected.

