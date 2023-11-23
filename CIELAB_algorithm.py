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


def blob_op(image, image_out):
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
    im_with_keypoints = cv2.drawKeypoints(image_out, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints


def nothing(x):
    pass


cv2.namedWindow("Threshold")
cv2.createTrackbar("Threshold a", "Threshold", 122, 255, nothing)
cv2.createTrackbar("Threshold b", "Threshold", 125, 255, nothing)


def lab_algorithm(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    a_thresh = cv2.getTrackbarPos("Threshold a", "Threshold")
    ret, a_picture = cv2.threshold(a, a_thresh, 255, cv2.THRESH_BINARY)
    a_pic = cv2.bitwise_not(a_picture)
    a_lined_image = draw_lines(a_pic)

    b_thresh = cv2.getTrackbarPos("Threshold b", "Threshold")
    ret, b_picture = cv2.threshold(b, b_thresh, 255, cv2.THRESH_BINARY)
    b_pic = b_picture
    b_lined_image = draw_lines(b_pic)
    # cv2.imshow('CIELAB-Binary', a_lined_image)
    # cv2.imshow('CIELAB-RGB', blob_op(a_lined_image, image))
    # cv2.imshow('CIELAB-Lab - a', blob_op(a_lined_image, a_lined_image))
    # cv2.imshow('CIELAB-Lab - b', blob_op(b_lined_image, b_lined_image))
    img = cv2.bitwise_and(a_lined_image, b_lined_image)
    cv2.imshow('CIELAB-Lab - bitwised', blob_op(img, image))


while i < t_frames:
    picture = os.listdir('./Frames')[i]
    img = ('Frames/' + str(picture))
    #img = ('Frames/' + 'Frame370.png')
    pictures.append(img)

    curr_img = cv2.imread(img)
    dst = cv2.undistort(curr_img, CM, DIST, None, NCM)
    out = cv2.warpPerspective(dst, M, (curr_img.shape[1], curr_img.shape[0]))
    lab_algorithm(out)

    cv2.waitKey(10)
    i += 1
