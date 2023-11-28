import cv2
import numpy as np


def blobweeds(binaryimage):
    keypoints = detector.detect(ExG_inv)
    current = []  # Empty current on every frame.
    if len(keypoints) != 0:  # If there are any BLOBs - Sanity check
        # Find the x and y coordiantes for the BLOBs in centimeters.
        for objs in range(len(keypoints)):
            quadrant = keypoints[objs].pt[0]/PPcm_hor // 50
            x = (keypoints[objs].pt[0]+172)/PPcm_hor  # + ROI X
            y = (629-keypoints[objs].pt[1])/PPcm_ver  # + ROI Y
            current.append((x, y, quadrant))

        while len(current) > 0:
            x, y, quadrant = current.pop()
            if len(prior) > 0:
                unique = True
                for x_prior, y_prior in prior:
                    if np.abs(y_prior - y) <= cmpf * 1.5 and np.abs(x_prior - x) <= 17:   # cmpf * i%(3+1) 17 is a guess at error
                        unique = False
                if unique:
                    weeds.append((x, y, y/cmasec, quadrant))
            else:  # In case we don't have any priors to compare against, we just add the weeds.
                weeds.append((x, y, y/cmasec, quadrant))
        prior.clear()

        weeds.clear()  # Send weed data before this!

        # Add the currently seen weeds to a list, as the prior weeds.
        for objs in range(len(keypoints)):
            x = (keypoints[objs].pt[0] + 172) / PPcm_hor  # + ROI X
            y = (629 - keypoints[objs].pt[1]) / PPcm_ver  # + ROI Y
            prior.append((x, y))

    blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('huh', blobs)  # Nothing will show, unless you feed the function a video
    cv2.waitKey(1)


# We need to define a BLOB detector
params = cv2.SimpleBlobDetector.Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 100
params.maxArea = 2500

# Ignore the remaining parameters.
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

# Find the blobs with the detector
detector = cv2.SimpleBlobDetector.create(params)

# We define the remaining stuff we need for keeping track
weeds = []
prior = []

# Important stuff for calculating weeds movement in a picture!
kmh = 5
meterasec = (kmh*1000)/3600
cmasec = meterasec*100

fps = 30  # This can't be defined for playback, so let's assume 30?
cmpf = meterasec/fps * 100  # Centimeters per frame

PPcm_ver = 4.2  # 934 pixels (1106-172) / 200 centimeteres
PPcm_hor = 4.67  # 105 pixels (196-91) / 25 centimeteres

# Write the code for taking an image, and converting to binary, then send it to BLOB

