import cv2
import numpy as np


def blobweeds(binaryimage, i):
    if i % 3 == 0:  # This is done purely for framerate, BLOB detection is not a fast algorithm
        keypoints = detector.detect(binaryimage)
        current = []  # Empty current on every frame.
        if len(keypoints) != 0:  # If there are any BLOBs - Sanity check
            # Find the x and y coordiantes for the BLOBs in centimeters.
            for objs in range(len(keypoints)):
                x = (keypoints[objs].pt[0] + 162) / PPcm_hor  # + ROI X
                y = (629 - keypoints[objs].pt[1]) / PPcm_ver  # + ROI Y
                size = keypoints[objs].pt[2]
                current.append((x, y, size))

            while len(current) > 0:
                x, y, size = current.pop()

                if len(prior) > 0:
                    unique = True
                    for x_prior, y_prior in prior:
                        if np.abs(y_prior-y) <= cmpf * 4 and np.abs(x_prior-x) <= 17:  # cmpf * i%(3+1) 17 is a guess at error
                            unique = False
                        if y >= 125:  # We only want the weeds at the top of the image, consider changing to be dependt on speed.
                            unique = False
                    if unique:
                        weeds.append((x, y, y/cmasec))  # Add size if wanted
                else:  # In case we don't have any priors to compare against, we just add the weeds.
                    weeds.append((x, y, y/cmasec))  # Add size if wanted
            prior.clear()
            weeds.clear()  # Send weed data before this!

            for objs in range(len(keypoints)):
                x = (keypoints[objs].pt[0] + 162) / PPcm_hor  # + ROI X
                y = (629 - keypoints[objs].pt[1]) / PPcm_ver  # + ROI Y
                prior.append((x, y))


# We need to define a BLOB detector
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

# We define the remaining stuff we need for keeping track
i = 0
weeds = []
prior = []

# Important stuff for calculating weeds movement in a picture!
kmh = 5
meterasec = (kmh*1000)/3600
cmasec = meterasec*100

fps = 30  # This can't be defined for playback, so let's assume 30?

cmpf = meterasec/fps * 100  # Centimeters per frame
PPI_ver = 720/57.480315  # height in pixels / height in inches
PPcm_ver = PPI_ver/2.54  # Converting to centimeters

PPI_hor = 1280/94.488189  # width in pixels / width in inches
PPcm_hor = PPI_hor/2.54  # Converting to centimeters

# Write the code for taking an image, and converting to binary, then send it to BLOB

