import cv2
import numpy as np
from picamera2 import Picamera2


Rot = np.load('BirdRot.npz')
M = Rot['BirdRot']

Cali = np.load('CalibratedCameraData/ArduCam.npz')  # Change this dependent on camera
CM = Cali['CamMatrix']
NCM = Cali['NewCamMatrix']
DIST = Cali['Distortion']


# Of course, we now have a bunch of images with blobs, let's find the big ones!
params = cv2.SimpleBlobDetector.Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 200

# Ignore the remaining parameters.
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

# Find the blobs with the detector
detector = cv2.SimpleBlobDetector.create(params)

i = 0
weeds = []


def blob_op(image, rgb):
	keypoints = detector.detect(image)
	'''
	if len(keypoints) != 0:
		for objs in range(len(keypoints)):
			x = keypoints[objs].pt[0]
			y = keypoints[objs].pt[1]
			print((x, y, int(y/weed_movement_second)))
			if y < 360:
				pass
				weeds.append((int(x), int(y), y/weed_movement_second))
				# print(weeds)
	'''
	blobs = cv2.drawKeypoints(rgb, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return blobs


# ExG algorithm
def exg(image, i):
	b, g, r = cv2.split(image)

	ExG = 2 * g - r - b

	# Let's try to use ExG
	ExG_threshold = cv2.inRange(ExG, np.array([35]), np.array([175]))
	ExG_blured = cv2.GaussianBlur(ExG_threshold, (3, 3), 0)
	close_ExG = cv2.morphologyEx(ExG_blured, cv2.MORPH_CLOSE, kernel=np.ones((5, 5)), iterations=2)
	ExG_inv = cv2.bitwise_not(close_ExG)
	cv2.imshow('ExG', ExG_inv)


# Lab algorithm
def lab_algorithm(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    ret, a_pic = cv2.threshold(a, 122, 255, cv2.THRESH_BINARY)
    a_picture = cv2.bitwise_not(a_pic)
	
    ret, b_picture = cv2.threshold(b, 125, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(a_picture, b_picture)
    cv2.imshow('video', img)
    return img


def hsv_algorithm(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (36, 45, 25), (86, 255, 255))

    imask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]

    hue, sat, val = cv2.split(green)
    ret, pic = cv2.threshold(sat, 40, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(pic, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


# ExHSV algorithm
def exhsv(image):
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    R, G, B = cv2.split(image_RGB) 

    ExG = 255 * (2 * G - R - B)
    

    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(image_HSV) 

    threshold = cv2.inRange(S, 55, 200)
    img_out = ExG & threshold
    threshold2 = cv2.inRange(img_out, min_threshold, max_threshold)
    close_ExG = cv2.morphologyEx(threshold2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return close_ExG

# Important stuff for calculating weeds movement in a picture!
meterasec = 1.39  # Redefine dependent on speed (5 kmh to 1.39 m/s)
fps = 30  # This can't be defined for playback, so let's assume 30?
cmpf = meterasec/fps * 100  # Centimeters per frame
PPI_vertical = 720/42.7165  # height in inches
PPcm_vertical = PPI_vertical/2.54
weed_movement_second = PPcm_vertical*cmpf # How many pixels does a weed move in a second?

with Picamera2() as cam:
    w, h = 1280, 720
    cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (w, h)}))
    cam.start()
    img_counter = 0
    while True:
        curr_img = cam.capture_array()
        out = cv2.warpPerspective(curr_img, M, (curr_img.shape[1], curr_img.shape[0]))
        resized_image = out[91:312, 162:int(1116)]
        #exg(resized_image, i)
        #lab_algorithm(resized_image)
        #hsv_algorithm(resized_image)
        #exhsv(resized_image)
        
        cv2.waitKey(1)

