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
    #print(picture)
    img = ('Frames/' + str(picture))
    pictures.append(img)
    i += 1 
    
i=0
sorted_pics = np.sort(pictures)

while i < t_frames:
    picture = sorted_pics[i+1]
    curr_img = cv2.imread(picture)
    image = cv2.undistort(curr_img, CM, DIST, None, NCM)
    
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    R, G, B = cv2.split(image)

    ExG = 2 * G - R - B

    mergred = cv2.merge([B, G, R])

    image_HSV = cv2.cvtColor(mergred, cv2.COLOR_BGR2HSV)

    H, S, V = cv2.split(image_HSV) 

    min_threshold = 35
    max_threshold = 175

    ExG_threshold = cv2.inRange(S, min_threshold, max_threshold)
    #ret, ExG_threshold = cv2.threshold(S, 40, 255, cv2.THRESH_BINARY)
    ExG_blured = cv2.GaussianBlur(ExG_threshold, (3, 3), 0)
    close_ExG = cv2.morphologyEx(ExG_blured, cv2.MORPH_CLOSE, kernel=np.ones((5, 5)), iterations=2)
    ExG_inv = cv2.bitwise_not(close_ExG) 
    
    cv2.imshow('Undistored', ExG_inv)
    out = cv2.warpPerspective(ExG_inv, M, (curr_img.shape[1], curr_img.shape[0]))
    cv2.imshow('Bird', out)

    cv2.waitKey(1)
    i += 1 
    