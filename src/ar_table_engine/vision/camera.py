import cv2 as cv

from ar_table_engine.utils.platform_info import CURRENT_OS, OS


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
        if key == ord("q"):
            break
        time.sleep(5)
    cap.release()
