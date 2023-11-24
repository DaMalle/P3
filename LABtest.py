import cv2
import numpy as np
from picamera2 import Picamera2

Rot = np.load('BirdRot.npz')
M = Rot['BirdRot']

Cali = np.load('CalibratedCameraData/ArduCam.npz')  # Change this dependent on camera
CM = Cali['CamMatrix']
NCM = Cali['NewCamMatrix']
DIST = Cali['Distortion']


def blob_op(image, image_out):
    params = cv2.SimpleBlobDetector.Params()
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 99999
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector.create(params)
    invert_image = cv2.bitwise_not(image)
    keypoints = detector.detect(invert_image)
    im_with_keypoints = cv2.drawKeypoints(image_out, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints
    

def lab_algorithm(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    ret, a_pic = cv2.threshold(a, 122, 255, cv2.THRESH_BINARY)
    a_picture = cv2.bitwise_not(a_pic)
	
    ret, b_picture = cv2.threshold(b, 125, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(a_picture, b_picture)
    #cv2.imshow('CIELAB-Lab - bitwised', blob_op(img, image))
    return blob_op(img, image)


with Picamera2() as cam:
	w, h = 1280, 720
	cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (w, h)}))
	cam.start()
	
	img_counter = 0
	while True:
		frame = cam.capture_array()
		out = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
		resized_image = out[91:312, 162:int(1116)]
		
		cv2.imshow("window", lab_algorithm(resized_image))
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

 
