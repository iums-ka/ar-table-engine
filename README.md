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

See [src layout vs. flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) for further details.

```text
ar-table-engine/
├── pyproject.toml
├── README.md
├── .gitignore
├── sitecustomize.py
│
├── data/
│   ├── config.json
│   └── calibration/                 # Camera intrinsics and homography
│
├── native/
│   └── aruco_nano/
│       ├── aruco_nano.h             # Native ArUco Nano implementation
│       ├── python_bindings.cpp      # pybind11 bindings
│       └── CMakeLists.txt           # Native extension build configuration
│
├── scripts/
│   └── build_native.py              # Builds the native Python extension
│
├── src/
│   └── ar_table_engine/
│       ├── __init__.py
│       ├── tasks/                   # Calibration, detection, WebSocket broadcast
│       ├── utils/
│       ├── vision/                  # ArUco detection and other vision components
│       │   └── aruco/
│       │       ├── detection.py
│       │       └── aruco_nano_ext.cp312-win_amd64.pyd
│       └── ws/
│
├── build/                           # Generated native build files
│
└── tests/
    └── ...
```


## Requirements

### Software

- **Python 3.12.10** (Python `>=3.12.10,<3.13` is supported).
- **[uv](https://docs.astral.sh/uv/)** for Python environment and dependency management.
- **CMake 3.20 or newer** for building the native ArUco Nano extension.
- **C++ compiler:** [Visual Studio Community](https://visualstudio.microsoft.com/de/vs/community/) on Windows, with the _Desktop development with C++_ workload installed, including MSVC and a Windows SDK.
- **[OpenCV 4.11](https://opencv.org/releases/)** native C++ libraries and development files, including headers, libraries, and runtime DLLs. The native extension must be built against a compatible OpenCV installation.

The Python OpenCV dependency (`opencv-python==4.11.0.86`) is installed automatically by `uv`.

### Hardware

- **Camera:** A Logitech C925e, Logitech Brio, or another camera that works reliably with OpenCV. A stable frame rate and sufficient image resolution are recommended.
- **Projector:** Required for projector-camera calibration and projected marker detection.
- **Display setup:** A monitor and projector configuration suitable for the intended table and projection area.

## Installation

### 1. Clone the repository

Clone the repository and open a terminal in the project root.

### 2. Install Python dependencies

```bash
uv sync --extra dev
```

This creates or updates the project environment and installs the runtime and development dependencies, including CMake, pybind11, pytest, and Ruff.

### 3. Configure the native OpenCV installation

The ArUco Nano Python extension uses native OpenCV C++ libraries. Install or build a compatible OpenCV 4.11 C++ distribution and configure the `OPENCV_ROOT` environment variable to point to its root directory.

For example, on Windows, the directory should contain an OpenCV `bin` directory with runtime DLLs and a `lib` directory containing `OpenCVConfig.cmake` and the required libraries.

Create a local `.env` file in the project root and set:

```dotenv
OPENCV_ROOT=C:/path/to/opencv
```

Replace the example path with the location of your OpenCV installation. Keep `.env` local and do not commit machine-specific paths.

### 4. Build the native ArUco Nano extension

From the project root, run:

```bash
uv run python scripts/build_native.py
```

This configures and builds the native extension using the configured OpenCV installation.

If you change the OpenCV version or installation, rebuild the extension against the corresponding native libraries.

### 5. Verify the installation

Run the test suite:

```bash
uv run pytest
```

You can also verify that the native extension imports successfully:

```bash
uv run python -c "from ar_table_engine.vision.aruco import aruco_nano_ext; print('ArUco Nano extension loaded successfully')"
```







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
{
  "aruco_detection": {
    "physical_marker_dict": "DICT_4X4_250",
    "projected_marker_dict": "DICT_5X5_250",
    "detector_parameters": {
      "error_correction_rate": 0.0,
      "min_size": 10,
      "max_attempts_per_candidate": 5,
      "detect_inverted_marker": false
    }
  }
}
```

- **`physical_marker_dict`**  
  Dictionary used for **real, printed markers** detected by the camera.

- **`projected_marker_dict`**  
  Dictionary used for **markers projected onto the table**, for example during calibration.

  ⚠️ **Use different dictionaries** for physical and projected markers to reduce ambiguity and prevent one type from being mistaken for the other.

- **`detector_parameters`** _(optional)_  
  Optional fine-tuning parameters for the ArUco Nano detector. If omitted, the native detector defaults are used.

  | Parameter | Description |
  |---|---|
  | `error_correction_rate` | Controls the amount of bit-error correction permitted during marker identification. Higher values may improve detection of imperfect markers but can increase false positives. |
  | `min_size` | Minimum contour side length for a candidate marker. Lower values can help detect smaller markers; higher values may reject noise but miss distant markers. |
  | `max_attempts_per_candidate` | Maximum number of attempts to identify a candidate while slightly adjusting its corners. More attempts may improve detection at additional computational cost. |
  | `detect_inverted_marker` | Enables detection of inverted markers, such as light markers on a dark background. Keep `false` for conventional dark markers on a light background. |

  These parameters can generally be left at their defaults initially and tuned if detection reliability or performance needs improvement. Parameter names must match the names exposed by the Python binding, using `snake_case`.

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
      "id": 12,
      "x": 560,
      "y": 726,
      "orientation": [0.56, 0.71]
    },
    {
      "Id": 7,
      "x": 2103,
      "y": 302,
      "orientation": [-0.13, 0.34]
    }
  ]
}
```

* **id**: The marker's ID within the configured ArUco dictionary.
* **x / y**: The marker's center position in the canonical coordinate system, with the origin at the top-left corner.
* **orientation**: A two-dimensional orientation vector [x, y] expressed in the canonical coordinate system. Each component is in the range [-1, 1].

---

## Notes

* Calibration and detection **must use the same camera resolution**
* Ensure the camera index corresponds with the intended camera 

