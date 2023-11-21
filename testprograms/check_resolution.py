import time
import cv2 as cv
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder


with Picamera2() as cam:
    cam.configure(cam.create_preview_configuration(main={"size": (2028, 1080),"format": "RGB888"}))
    encoder = H264Encoder(10000000)
    cam.start_recording(encoder, "please_be_4k.h264")
    #cam.start_preview(Preview.QTGL)
    time.sleep(5)
    #cam.stop_preview()
    cam.stop_recording()
