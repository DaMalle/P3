import cv2
import numpy as np

image = cv2.imread('F3.png')

image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

R, G, B = cv2.split(image)

ExG = 2 * G - R - B

mergred = cv2.merge([B, G, R])

image_HSV = cv2.cvtColor(mergred, cv2.COLOR_BGR2HSV)

H, S, V = cv2.split(image_HSV) 

min_threshold = 55
max_threshold = 215

ExG_threshold = cv2.inRange(S, min_threshold, max_threshold)
ExG_blured = cv2.GaussianBlur(ExG_threshold, (3, 3), 0)
close_ExG = cv2.morphologyEx(ExG_blured, cv2.MORPH_CLOSE, kernel=np.ones((5, 5)), iterations=2)
ExG_inv = cv2.bitwise_not(close_ExG)

cv2.imshow('Original image', image)
cv2.imshow('ExG_HSV image', ExG_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

