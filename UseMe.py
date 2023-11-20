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

while i < t_frames:
    picture = os.listdir('./Frames')[i]
    curr_img = cv2.imread(('Frames/' + str(picture)))
    dst = cv2.undistort(curr_img, CM, DIST, None, NCM)
    out = cv2.warpPerspective(dst, M, (curr_img.shape[1], curr_img.shape[0]))

    cv2.imshow('Bird', out)
    cv2.waitKey(1)
    i += 1
