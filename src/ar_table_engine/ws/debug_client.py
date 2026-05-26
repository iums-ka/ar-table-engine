import asyncio
import json

import cv2 as cv
import numpy as np
import websockets

from ar_table_engine.utils.file_utils import load_config

SERVER_URI = "ws://localhost:5001"


async def run(test_img):
    print(f"Connecting to {SERVER_URI}...")
    async with websockets.connect(SERVER_URI) as websocket:
        print("Connected. Waiting for messages...\n")

        try:
            async for message in websocket:
                print("Received:", message)

                # Parse JSON message
                data = json.loads(message)

                # Clone base image so we redraw fresh every frame
                frame = test_img.copy()
                if "markers" in data:
                    for marker in data["markers"]:
                        marker_data = marker.get("Data", {})
                        x = int(marker_data.get("X", 0))
                        y = int(marker_data.get("Y", 0))

                        # Draw red dot (BGR: 0,0,255)
                        cv.circle(
                            frame, (x, y), radius=10, color=(0, 0, 255), thickness=-1
                        )
                # Show updated image
                cv.imshow("Markers", cv.resize(frame, None, fx=0.3, fy=0.3))
                cv.waitKey(1)

        except websockets.ConnectionClosed:
            print("Connection closed")


if __name__ == "__main__":
    CFG = load_config(r"./data/config.json")
    test_img = cv.imread(r"./data/text-test-4k.png")
    cam_to_proj_H = np.load(r"./data/calibration/cam_to_proj_H.npy")
    bounding_box_H = np.load(r"./data/calibration/bounding_box_H.npy")
    WNAME = "Test-Client"
    WINDOW = cv.namedWindow(WNAME, cv.WINDOW_NORMAL)
    cv.moveWindow(
        WNAME,
        x=CFG["projector"]["screen_position"][0],
        y=CFG["projector"]["screen_position"][1],
    )
    cv.setWindowProperty(WNAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow(WNAME, test_img)
    cv.waitKey(1)
    asyncio.run(run(test_img))
