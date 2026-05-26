# AR Table Engine

This project includes AR Table (camera-projector-system) calibration logic (encompassing calibration result persistence) and service logic to broadcast detection results to clients using a given calibration.

## Responsibilities

### What the engine does

- Camera calibration (intrinsics, distortion correction)
- Projector ↔ camera alignment (homography mapping)
- Real-time ArUco marker detection and tracking
- State stabilization and tracking lifecycle management
- Streaming spatial data via WebSocket

### What the engine does not do
It does not contain any application-specific logic or user-facing features. Instead, it exposes raw spatial information (e.g. position, rotation (WIP)) to downstream tasks such as frontend applications that further interpret these results to enable meaningful interactions.

## Related Repositories

- **[ar-table-apps](https://github.com/iums-ka/ar-table-apps)**  
  Collection of applications built on top of the engine. These apps consume the spatial data stream and implement user-facing experiences, interactions, and domain-specific logic.


## Project Structure (Relevant Parts Only)
See [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) for further details.
```

ar-table-engine/
├── pyproject.toml
├── README.md
├── .gitignore
├── sitecustomize.py
│
├── data/
│   ├── config.json
│   └── calibration/                 # Intrinsics and homography
│
├── src/
│   └── ar_table_engine/
│       ├── __init__.py
│       ├── tasks/                   # Calibration, detection, websocket broadcast          
│       ├── utils/
│       ├── vision/                  # ArUco and more
│       └── ws/
│
└── tests/
    └── ...

```

---

## Requirements

- Python ≥ 3.12.10
- Optimally, a Logitech C925e, Logitech Brio or other camera that works well with OpenCV
- Projector (for calibration)

---

## Installation

From the project root:

```bash
uv sync --extra dev  # includes pytest and such 
```

---

## Configuration (`config.json`)

Before running **calibration** or **detection**, you must configure `config.json`.

### Camera

```json
"camera": {
    "width": 1920,
    "height": 1080,
    "index": 1,
    "fps": 30
}
```

* **width / height**
  Resolution the camera is running at. This must match the calibration resolution.

* **index**
  OpenCV camera index passed to `cv.VideoCapture(index)`.

  * `0` = default camera
  * `1`, `2`, … = additional connected cameras
    If the wrong camera opens, this value usually needs to change.

* **fps**
  Requested camera framerate.

---

### Projector

```json
"projector": {
    "width": 3840,
    "height": 2160,
    "screen_position": [1920, 0]
}
```

* **width / height**
  Resolution of the projector display.

* **screen_position**
  The **top-left corner of the projector in global screen coordinates**.
  This is critical during calibration so that projected calibration images appear on the correct display.

  Example:

  * Main display: `1920×1080`
  * Projector placed to the **right**
  * Projector top-left → `[1920, 0]`

---

### Main Display

```json
"main_display": {
    "width": 1920,
    "height": 1080
}
```

Resolution of the primary monitor. Used for correct window placement.

---

### ArUco Detection

```json
"aruco_detection": {
    "physical_marker_dict": "DICT_4X4_250",
    "projected_marker_dict": "DICT_5X5_250",
    "detector_parameters": {
        "perspectiveRemoveIgnoredMarginPerCell": 0.1,
        "perspectiveRemovePixelPerCell": 4,
        "errorCorrectionRate": 0.5
    }
}
```

* **physical_marker_dict**
  Dictionary used for **real, printed markers** visible to the camera.

* **projected_marker_dict**
  Dictionary used for **markers projected onto the table** during calibration.

  ⚠️ **These should be different dictionaries** to avoid false detections and ambiguity.

* **detector_parameters**
  Fine-tuning parameters for OpenCV’s ArUco detector. These can generally be ignored unless detection issues arise.

---

## Calibration

Before running detection, **calibration files must exist** in the `ar_table_engine/calibration` directory.

Required files:

* `undistortion_args.npz`
* `cam_to_proj_H.npy`
* `bounding_box_H.npy`

The two homography files (two latter files above) can either be placed manually into the directory or generated automatically. This project does not offer functionality to generate undistortion_args.npz. This file can be created using [this script](https://github.com/iums-ka/ar-table-calibrator-with-undistortion/blob/main/ar_table_calibrator/calibration/camera_undistortion.py).

### Run calibration

From the project root:

```bash
uv run python -m ar_table_engine.tasks.calibration
# or with active venv, simply:
python -m ar_table_engine.tasks.calibration
```

> Note: The undistortion arguments file must be manually placed into the calibration directory, as it is not generated within this calibration.

---

## Detection

Once all required calibration files are present, marker detection can be started.

From the project root:

```bash
uv run python -m ar_table_engine.tasks.detection
# or with active venv, simply:
python -m ar_table_engine.tasks.detection
```

This starts a WebSocket server on:

```
ws://localhost:5001
```

---

## WebSocket Output Format

Detected markers are broadcast as JSON messages in the following format:

```json
{
  "markers": [
    {
      "Id": 12,
      "MessageType": "CONTROLHOVER",
      "Data": {
        "X": 560,
        "Y": 726
      }
    },
    {
      "Id": 7,
      "MessageType": "CONTROLHOVER",
      "Data": {
        "X": 2103,
        "Y": 302,
      }
    }
  ]
}
```

* **Id**: Marker ID
* **X / Y**: Marker center position in projector screen space (origin in top left) - in the future we will switch to normalized coordinates.
* **MessageType**: Fixed message type for downstream consumers

---

## Notes

* Calibration and detection **must use the same camera resolution**
* Ensure the camera index corresponds with the intended camera 

