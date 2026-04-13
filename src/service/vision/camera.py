from service.utils.platform_info import CURRENT_OS, OS
from enum import Enum, auto
import cv2 as cv
import numpy as np

def init_video_capture(cam_idx, width, height, fps):
    if CURRENT_OS is OS.LINUX:
        vid_api = cv.CAP_V4L2
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        # fourcc = None
    elif CURRENT_OS is OS.WINDOWS:
        vid_api = cv.CAP_MSMF
        fourcc = None
    else:
        vid_api = cv.CAP_ANY
        fourcc = None

    cap = cv.VideoCapture(cam_idx, vid_api)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    if fourcc is not None:
        cap.set(cv.CAP_PROP_FOURCC, fourcc)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)
    cap.set(cv.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap.set(cv.CAP_PROP_FOCUS, 0)  # Focus on infinity
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    return cap


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the captured image to improve marker detection for cases where markers are not being recognized reliably.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    clahe_img = cv.GaussianBlur(clahe_img, (5, 5), 0)

    adaptive = cv.adaptiveThreshold(clahe_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 65, 2)
    _, otsu = cv.threshold(adaptive, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    combined = cv.bitwise_and(otsu, adaptive)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    cleaned = cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel)

    cleaned = cv.cvtColor(cleaned, cv.COLOR_GRAY2BGR)

    # return clahe_img
    return cleaned



if __name__ == "__main__":
    cap = init_video_capture(0, 1920, 1080, 30)
    print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv.CAP_PROP_FPS))
    print(cap.get(cv.CAP_PROP_BUFFERSIZE))
    
    # For testing camera frame buffer
    import time
    while True:
        ret, frame = cap.read()
        print(frame.shape)
        if not ret:
            print("No camera found")
            break
        # frame_flipped = np.flip(frame, axis=1)
        cv.imshow("name", cv.resize(frame, None, fx=0.5, fy=0.5))
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        time.sleep(5)
    cap.release()