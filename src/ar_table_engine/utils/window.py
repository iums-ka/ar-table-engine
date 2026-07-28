import uuid

import cv2 as cv
import numpy as np


class Window:
    def __init__(self, name: str):
        self.name = f"{name}-{uuid.uuid4().hex}"
        cv.namedWindow(self.name, cv.WINDOW_NORMAL)

    def show(self, frame):
        # Resize while keeping aspect ratio
        # Size of drawable area in window
        x, y, max_w, max_h = cv.getWindowImageRect(self.name)

        if max_w > 0 and max_h > 0:
            h, w = frame.shape[:2]

            # Resize to fit within drawable area
            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)

            # Center image in drawable area to prevent stretching due to cv.WINDOW_NORMAL
            canvas = np.zeros((max_h, max_w, 3), dtype=frame.dtype)
            x_off = (max_w - new_w) // 2
            y_off = (max_h - new_h) // 2
            canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

            frame = canvas

        cv.imshow(self.name, frame)

    def set_position(self, x: int, y: int):
        cv.moveWindow(self.name, x, y)

    def set_fullscreen(self, enable: bool):
        if enable:
            cv.setWindowProperty(
                self.name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN
            )
        else:
            cv.setWindowProperty(self.name, cv.WND_PROP_FULLSCREEN, 0)
