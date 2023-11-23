import cv2
import numpy as np
import os

def exg(image, i):
    b, g, r = cv2.split(image)

    ExG = 2 * g - r - b

    """""
    output_full = 2 * (g / (r + g + b)) - (r / (r + g + b)) - (b / (b + g + r))

    b_c = np.clip(b, 1, 256)
    g_c = np.clip(g, 1, 256)
    r_c = np.clip(r, 1, 256)
    output_half = 2 * (g_c / (r_c + g_c + b_c)) - (r_c / (r_c + g_c + b_c)) - (b_c / (b_c + g_c + r_c))
    """""

    # Let's try to use ExG
    ExG_threshold = cv2.inRange(ExG, np.array([35]), np.array([175]))
    ExG_blured = cv2.GaussianBlur(ExG_threshold, (3, 3), 0)
    close_ExG = cv2.morphologyEx(ExG_blured, cv2.MORPH_CLOSE, kernel=np.ones((5, 5)), iterations=2)
    ExG_inv = cv2.bitwise_not(close_ExG)

    if i % 2.5 == 0:
        keypoints = detector.detect(ExG_inv)
        if len(keypoints) != 0:
            for objs in range(len(keypoints)):
                x = keypoints[objs].pt[0]
                y = keypoints[objs].pt[1]
                print((x, y, int(y/weed_movement_second)))
                if y < 360:
                    pass
                    weeds.append((int(x), int(y), y/weed_movement_second))
                    # print(weeds)


            print(" ")
        blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('huh', blobs)
    cv2.imshow('video', image)


Rot = (np.load('BirdView\\BirdRot.npz'))
M = Rot['BirdRot']

Cali = np.load('CamCali\\HQCam.npz')  # Change this dependent on camera
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

t_frames = len(os.listdir('./Frames'))
i = 0
pictures = []
weeds = []

# Important stuff for calculating weeds movement in a picture!
meterasec = 1.39  # Redefine dependent on speed (5 kmh to 1.39 m/s)
fps = 30  # This can't be defined for playback, so let's assume 30?
cmpf = meterasec/fps * 100  # Centimeters per frame
PPI_vertical = 720/42.7165  # height in inches
PPcm_vertical = PPI_vertical/2.54
weed_movement_second = PPcm_vertical*cmpf # How many pixels does a weed move in a second?

while i < t_frames:
    picture = os.listdir('./Frames')[i]
    curr_img = cv2.imread(('Frames/' + str(picture)))
    # cv2.imshow('lab', cv2.cvtColor(curr_img, cv2.COLOR_BGR2LAB))
    dst = cv2.undistort(curr_img, CM, DIST, None, NCM)
    out = cv2.warpPerspective(dst, M, (curr_img.shape[1], curr_img.shape[0]))
    pictures.append(out)
    exg(out, i)
    cv2.waitKey(1)
    i += 1
