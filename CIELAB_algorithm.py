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


def draw_lines(image):
    pic = image
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


def lab_algorithm(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    ret, picture = cv2.threshold(a, 120, 255, cv2.THRESH_BINARY)
    pic = cv2.bitwise_not(picture)
    lined_image = draw_lines(pic)
    #cv2.imshow('CIELAB-Binary', lined_image)
    cv2.imshow('CIELAB-RGB', blob_op(lined_image, image))
    #cv2.imshow('CIELAB-Lab', blob_op(lined_image, lab))


while i < t_frames:
    picture = os.listdir('./Frames')[i]
    img = ('Frames/' + str(picture))
    pictures.append(img)

    curr_img = cv2.imread(img)
    dst = cv2.undistort(curr_img, CM, DIST, None, NCM)
    out = cv2.warpPerspective(dst, M, (curr_img.shape[1], curr_img.shape[0]))
    lab_algorithm(out)

    cv2.waitKey(10)
    i += 1