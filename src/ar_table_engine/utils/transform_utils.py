import cv2 as cv


def dist_to_map(camMtx, distCoeffs, camMtxNew, w, h):
    map_a, map_b = cv.initUndistortRectifyMap(
        camMtx, distCoeffs, None, camMtxNew, (w, h), cv.CV_16SC2
    )
    return map_a, map_b
