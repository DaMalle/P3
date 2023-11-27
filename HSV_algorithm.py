import cv2
import numpy as np
import os

Rot = (np.load('BirdView\\BirdRot.npz'))
M = Rot['BirdRot']

Cali = np.load('CamCali\\HQCam.npz')  # Change this dependent on camera
CM = Cali['CamMatrix']
NCM = Cali['NewCamMatrix']
DIST = Cali['Distortion']

t_frames = len(os.listdir('./Frames'))
i = 0
pictures = []


def draw_lines(image, rgb_):
    pic = morphology_op(image, rgb_)
    pt_a = [155, 95]
    pt_b = [1145, 22]
    pt_c = [355, 717]
    pt_d = [970, 680]

    color = (255, 255, 255)

    cv2.line(pic, (pt_a[0], pt_a[1]), (pt_b[0], pt_b[1]), color, 2)
    cv2.line(pic, (pt_c[0], pt_c[1]), (pt_d[0], pt_d[1]), color, 2)
    cv2.line(pic, (pt_a[0], pt_a[1]), (pt_c[0], pt_c[1]), color, 2)
    cv2.line(pic, (pt_b[0], pt_b[1]), (pt_d[0], pt_d[1]), color, 2)
    return pic


def blob_op(image, rgb_):
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
    im_with_keypoints = cv2.drawKeypoints(rgb_, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints


def morphology_op(image, rgb_):
    #kernel = np.ones((8, 8), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    np.array([[0, 0, 0, 1, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 0],
              [1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    blobbed_image = blob_op(closing, rgb_)
    return blobbed_image


hueMin = 36
hueMax = 86
saturationMin = 45
saturationMax = 255
brightnessMin = 25
brightnessMax = 255


def hsv_algorithm(image, rgb_):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (hueMin, saturationMin, brightnessMin), (hueMax, saturationMax, brightnessMax))

    imask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]

    hue, sat, val = cv2.split(green)
    ret, pic = cv2.threshold(sat, 40, 255, cv2.THRESH_BINARY)

    cv2.imshow('HSV', draw_lines(pic, rgb_))


while i < t_frames:
    picture = os.listdir('./Frames')[i]
    img = ('Frames/' + str(picture))
    pictures.append(img)

    curr_img = cv2.imread(img)
    dst = cv2.undistort(curr_img, CM, DIST, None, NCM)
    #cv2.imshow('Undistorted', dst)
    out = cv2.warpPerspective(dst, M, (curr_img.shape[1], curr_img.shape[0]))
    #cv2.imshow('Bird', out)

    hsv_algorithm(out, out)

    cv2.waitKey(10)
    i += 1