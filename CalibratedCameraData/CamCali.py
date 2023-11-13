import numpy as np
import cv2 as cv
import glob
import numpy as np
from picamera2 import Picamera2

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./DataForArduCam/*.png')
print(images) 

for frame in images:
    img = cv.imread(frame)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (10,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (10,7), corners2, ret)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)
print(dist)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (img.shape[1], img.shape[0]), 0, (img.shape[1], img.shape[0]))
print(newcameramtx)
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# np.savez('ArduCam', CamMatrix=mtx, NewCamMatrix=newcameramtx, Distortion=dist)

with Picamera2() as cam:
	cam.configure(cam.create_preview_configuration(main={"format": "RGB888"}))
	cam.start()
	while True:
		frame = cam.capture_array()
		cv.imshow('before', frame)
		dst = cv.undistort(frame, mtx, dist, None, newcameramtx)

		cv.imshow("window", dst)
		cv.waitKey(1)
