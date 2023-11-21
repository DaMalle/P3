import time
import cv2 as cv
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

if __name__ == "__main__":
    Rot = np.load('BirdRotation.npz')
    Cali = np.load('CalibratedCameraData/HQCam.npz')
    BirdRotation = Rot['BirdRot']
    CM = Cali['CamMatrix']
    NCM = Cali['NewCamMatrix']
    DIST = Cali['Distortion']
    
    with Picamera2() as cam:
        w, h = 1280, 720
        config = cam.create_still_configuration(main={"format": "RGB888", "size": (w, h)})
        cam.configure(config)

        writer = cv.VideoWriter('outpy.h264',cv.VideoWriter_fourcc(*"x264"), 30, (w, h))
        cam.start()
        time.sleep(1)
        i = 100
        while i > 0:
            frame = cam.capture_array()
            dst = cv.undistort(frame, CM, DIST, None, NCM)
            out = cv.warpPerspective(dst, BirdRotation, (w, h))
            
            #cv.imshow('nodist', dst)
            cv.imshow("frame", out)
            cv.waitKey(1)
            i -= 1
        cv.imwrite('LastFrame.png', out)
        cam.stop()
