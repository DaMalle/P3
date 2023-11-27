import cv2
import numpy as np
from picamera2 import Picamera2

Rot = np.load('BirdRot.npz')
M = Rot['BirdRot']

Cali = np.load('CalibratedCameraData/ArduCam.npz')  # Change this dependent on camera
CM = Cali['CamMatrix']
NCM = Cali['NewCamMatrix']
DIST = Cali['Distortion']


with Picamera2() as cam:
	w, h = 1280, 720
	cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (w, h)}))
	cam.start()
	
	img_counter = 0
	while True:
		frame = cam.capture_array()
		#dst = cv2.undistort(frame, CM, DIST, None, NCM)
		out = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
		#cv2.line(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), (int(frame.shape[1]/2), int(frame.shape[0])), (0,0,255), 3)
		resized_image = out[91:312, 162:int(1116)]
		cv2.line(resized_image, (int(954/2),0), (int(954/2),221), (0,0,255), 2)
		cv2.line(resized_image, (int((954/2)/2),0), (int((954/2)/2),221), (0,0,255), 2)
		cv2.line(resized_image, (int((954/2)/2)+int(954/2),0), (int((954/2)/2)+int(954/2),221), (0,0,255), 2)
		cv2.imshow("window", frame)
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


"""
cv2.namedWindow("test")

img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
"""
