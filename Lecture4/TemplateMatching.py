import cv2
import numpy as np
import math

# read the input image as grayscale image
img = cv2.imread('neon-text.png', 0)
template = cv2.imread('neon-text_heart.png', 0)
print("Image data before Normalize:\n", img)

# Apply threshold to create a binary image
ret, thresh = cv2.threshold(img, 149, 255, cv2.THRESH_BINARY)
threst_template = cv2.threshold(img, 149, 255, cv2.THRESH_BINARY)
print("Image data after Thresholding:\n", thresh)

# visualize the normalized image
cv2.imshow("Threshold", img)
cv2.imshow("Threshold - Heart", template)
cv2.waitKey(0)
cv2.destroyAllWindows()
