import ctypes
import os
from pathlib import Path

from dotenv import load_dotenv

from ar_table_engine.utils.platform_info import CURRENT_OS, OS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load project configuration.
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

# Platform-specific configuration.
if CURRENT_OS is OS.WINDOWS:
    # Significantly faster VideoCapture initialization with MSMF.
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

    # Set DPI awareness for OpenCV windows.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass

elif CURRENT_OS is OS.LINUX:
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Register the OpenCV DLL directory on Windows.
# Keep the returned handle alive for the lifetime of this module.
_opencv_dll_directory = None

if CURRENT_OS is OS.WINDOWS:
    opencv_root_value = os.environ.get("OPENCV_ROOT")

    if opencv_root_value:
        opencv_root = Path(opencv_root_value).expanduser().resolve()
        opencv_bin = opencv_root / "bin"

        if opencv_bin.is_dir():
            _opencv_dll_directory = os.add_dll_directory(str(opencv_bin))
        else:
            print(f"Warning: OpenCV DLL directory does not exist: {opencv_bin}")
    else:
        print(f"Warning: OPENCV_ROOT is not set. Check {ENV_FILE}")
