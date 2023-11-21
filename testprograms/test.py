import time

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder


with Picamera2() as cam:
    config = cam.create_video_configuration()
    cam.configure(config)

    encoder = H264Encoder(10000000)
    #cam.start_recording(encoder, "noir_camera_test_2.h264")
    cam.start_preview()
    time.sleep(30)
    cam.stop_preview()
    #cam.stop_recording()

