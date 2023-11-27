import cv2
import numpy as np
from picamera2 import Picamera2

Rot = np.load('BirdRot.npz')
M = Rot['BirdRot']

Cali = np.load('CalibratedCameraData/ArduCam.npz')  # Change this dependent on camera
CM = Cali['CamMatrix']
NCM = Cali['NewCamMatrix']
DIST = Cali['Distortion']


def exhsv(image):
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    R, G, B = cv2.split(image_RGB) 

    ExG = 255 * (2 * G - R - B)
    

    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(image_HSV) 

    min_threshold = 55
    max_threshold = 200

    threshold = cv2.inRange(S, min_threshold, max_threshold)
    #ret, ExG_threshold = cv2.threshold(S, 40, 255, cv2.THRESH_BINARY)
    #ExG_blured = cv2.GaussianBlur(ExG_threshold, (3, 3), 0)
    #close_ExG = cv2.morphologyEx(ExG_threshold, cv2.MORPH_CLOSE, np.ones((5, 5)))
    #inv = cv2.bitwise_not(threshold)
    img_out = ExG & threshold
    threshold2 = cv2.inRange(img_out, min_threshold, max_threshold)
    close_ExG = cv2.morphologyEx(threshold2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return close_ExG


with Picamera2() as cam:
	w, h = 1280, 720
	cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (w, h)}))
	cam.start()
	
	img_counter = 0
	while True:
		frame = cam.capture_array()
		out = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
		resized_image = out[91:312, 162:int(1116)]
		
		cv2.imshow("window", exhsv(resized_image))
		cv2.waitKey(1)
		
		k = cv2.waitKey(1)
		if k%256 == 27:
			# ESC pressed
			print("Escape hit, closing...")
			break
		elif k%256 == 32:
			# SPACE pressed
			img_name = "opencv_frame2_{}.png".format(img_counter)
			cv2.imwrite(img_name, frame)
			print("{} written!".format(img_name))
			img_counter += 1
	cam.stop()
	cv2.destroyAllWindows()

 
