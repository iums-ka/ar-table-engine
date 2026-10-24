from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Project layout
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = PROJECT_ROOT / "native" / "aruco_nano"
BUILD_DIR = PROJECT_ROOT / "build" / "native" / "aruco_nano"


def run(command: list[str]) -> None:
    """Run a command and stop immediately if it fails."""
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def get_opencv_dir() -> Path:
    """Load and validate the OpenCV CMake package directory."""
    load_dotenv(PROJECT_ROOT / ".env")

    opencv_root_value = os.environ.get("OPENCV_ROOT")
    if not opencv_root_value:
        raise SystemExit(
            "OPENCV_ROOT is missing. Set it in the project-root .env file."
        )

    opencv_root = Path(opencv_root_value).expanduser().resolve()
    opencv_dir = opencv_root / "lib"

    if not (opencv_dir / "OpenCVConfig.cmake").is_file():
        raise SystemExit(f"Could not find OpenCVConfig.cmake: {opencv_dir}")
    return opencv_dir


def get_pybind11_dir() -> str:
    """Find pybind11's CMake package in the active Python environment."""
    result = subprocess.run(
        [sys.executable, "-m", "pybind11", "--cmakedir"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    pybind11_dir = result.stdout.strip()
    if not pybind11_dir:
        raise SystemExit(
            "Could not determine pybind11's CMake directory. "
            "Ensure pybind11 is installed in the active uv environment."
        )

    return pybind11_dir


def main() -> None:
    opencv_dir = get_opencv_dir()

    if not (NATIVE_SOURCE / "CMakeLists.txt").is_file():
        raise SystemExit(f"Native source directory is missing: {NATIVE_SOURCE}")

    pybind11_dir = get_pybind11_dir()

    # Configure the native build.
    run(
        [
            "cmake",
            "-S",
            str(NATIVE_SOURCE),
            "-B",
            str(BUILD_DIR),
            f"-DOpenCV_DIR={opencv_dir}",
            f"-Dpybind11_DIR={pybind11_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
        ]
    )

    # Compile the extension.
    run(
        [
            "cmake",
            "--build",
            str(BUILD_DIR),
            "--config",
            "Release",
            "--target",
            "aruco_nano_ext",
        ]
    )

    print("\nNative extension build completed successfully.")


if __name__ == "__main__":
    main()
