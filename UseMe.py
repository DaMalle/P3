import cv2
import numpy as np
import os

Rot = np.load('BirdRot.npz')
M = Rot['BirdRot']

Cali = np.load('HQCam.npz')  # Change this dependent on camera
CM = Cali['CamMatrix']
NCM = Cali['NewCamMatrix']
DIST = Cali['Distortion']

t_frames = len(os.listdir('./Frames'))
i = 0
pictures = []

while i < t_frames:
    picture = os.listdir('./Frames')[i]
    img = ('Frames/' + str(picture))
    curr_img = cv2.imread(img)
    dst = cv2.undistort(curr_img, CM, DIST, None, NCM)
    cv2.imshow('Undistored', dst)
    out = cv2.warpPerspective(dst, M, (curr_img.shape[1], curr_img.shape[0]))
    cv2.imshow('Bird', out)
    cv2.waitKey(10)
    i += 1
