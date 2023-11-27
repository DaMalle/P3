import cv2
import numpy as np
from picamera2 import Picamera2

Rot = np.load('BirdRot.npz')
M = Rot['BirdRot']

Cali = np.load('CalibratedCameraData/ArduCam.npz')  # Change this dependent on camera
CM = Cali['CamMatrix']
NCM = Cali['NewCamMatrix']
DIST = Cali['Distortion']


def blob_op(image, rgb):
    params = cv2.SimpleBlobDetector.Params()
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 3500
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector.create(params)
    invert_image = cv2.bitwise_not(image)
    keypoints = detector.detect(invert_image)
    im_with_keypoints = cv2.drawKeypoints(rgb, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints



def morphology_op(image, rgb):
    #kernel = np.ones((8, 8), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    blobbed_image = blob_op(closing, rgb)
    return blobbed_image


hueMin = 36
hueMax = 86
saturationMin = 45
saturationMax = 255
brightnessMin = 25
brightnessMax = 255


def hsv_algorithm(image, rgb):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (hueMin, saturationMin, brightnessMin), (hueMax, saturationMax, brightnessMax))

    imask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]

    hue, sat, val = cv2.split(green)
    ret, pic = cv2.threshold(sat, 40, 255, cv2.THRESH_BINARY)

    return morphology_op(pic, rgb)


with Picamera2() as cam:
	w, h = 1280, 720
	cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (w, h)}))
	cam.start()
	
	img_counter = 0
	while True:
		frame = cam.capture_array()
		out = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
		resized_image = out[91:312, 162:int(1116)]
		
		cv2.imshow("window", hsv_algorithm(resized_image, resized_image))
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

 
