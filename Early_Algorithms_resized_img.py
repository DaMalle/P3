import cv2
import numpy as np
import os


# ExG algorithm
def exg_algorithm(image):
    b, g, r = cv2.split(image)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    ExG = 2 * g - r - b

    imgOut = np.clip(ExG, 0, 255)
    imgOut = imgOut.astype(np.uint8)

    ExG_threshold = cv2.inRange(imgOut, np.array([30]), np.array([125]))
    close_ExG = cv2.morphologyEx(ExG_threshold, cv2.MORPH_CLOSE, kernel=np.ones((5, 5)), iterations=5)
    return close_ExG


# Lab algorithm
def lab_algorithm(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    ret, a_pic = cv2.threshold(a, 122, 255, cv2.THRESH_BINARY)
    a_picture = cv2.bitwise_not(a_pic)

    ret, b_picture = cv2.threshold(b, 125, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(a_picture, b_picture)
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


def labhsv_algorithm(image):
    # Lab
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    ret, a_pic = cv2.threshold(a, 122, 255, cv2.THRESH_BINARY)
    a_picture = cv2.bitwise_not(a_pic)
    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (36, 45, 25), (86, 255, 255))

    imask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]

    hue, sat, val = cv2.split(green)
    ret, green_hsv = cv2.threshold(sat, 40, 255, cv2.THRESH_BINARY)
    # Combine Lab & HSV
    threshold_result = a_picture | green_hsv

    return cv2.morphologyEx(
        cv2.morphologyEx(threshold_result, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))),
        cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))


Rot = np.load('BirdView/BirdRotArdu.npz')
M = Rot['BirdRot']

t_frames = len(os.listdir('./test5kmh'))
i = 0

params = cv2.SimpleBlobDetector.Params()

params.minThreshold = 0
params.maxThreshold = 255

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 25
params.maxArea = 5000

# Ignore the remaining parameters.
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.filterByColor = False

# Find the blobs with the detector
detector = cv2.SimpleBlobDetector.create(params)

while i < t_frames:
    picture = os.listdir('./test5kmh')[i]
    curr_img = cv2.imread(('test5kmh/' + str(picture)))
    resized_image = curr_img[91:312, 162:int(1116)]
    image = labhsv_algorithm(resized_image)

    keypoints = detector.detect(image)
    blobs = cv2.drawKeypoints(resized_image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('frame' + str(i+180), blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1
