import os

from ar_table_engine.utils.platform_info import CURRENT_OS, OS

# Video processing is very platform dependent - therefore we set up all relevant env-vars prior to any other script running
if CURRENT_OS is OS.WINDOWS:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = (
        "0"  # Significantly faster VideoCapture init for MSMF API
    )
    # Relevant for imshow (ignore Windows scaling): Set DPI Awareness (Windows 8, 10, 11)
    import ctypes

    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(1)
elif CURRENT_OS is OS.LINUX:
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
