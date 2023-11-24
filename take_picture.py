import cv2
import numpy as np
from picamera2 import Picamera2

Rot = np.load('BirdRot.npz')
M = Rot['BirdRot']

Cali = np.load('CalibratedCameraData/HQCam.npz')  # Change this dependent on camera
CM = Cali['CamMatrix']
NCM = Cali['NewCamMatrix']
DIST = Cali['Distortion']

with Picamera2() as cam:
    w, h = 1280, 720
    cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (w, h)}))
    cam.start()

    while True:
        frame = cam.capture_array()
        dst = cv2.undistort(frame, CM, DIST, None, NCM)
        out = cv2.warpPerspective(dst, M, (frame.shape[1], frame.shape[0]))
        cv2.imshow("window", out)
        cv2.waitKey(1)
    cam.stop()
