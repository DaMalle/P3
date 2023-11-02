import cv2
import numpy as np
import math


def apply_filter_grey(column, row):
    i = 0
    sum = 0
    while i < data2[0]:
        j = 0
        while j < data2[0]:
            sum = sum + (greyscaled_image[column+i, row+j]*mean_kernel[i][j])
            j = j + 1
        i = i + 1
    fin = sum/data3
    output_image[column, row] = fin


def apply_filter(column, row):
    temp = []
    temp2 = []
    i = 0
    while i < data2[0]:
        j = 0
        while j < data2[0]:
            k = 0
            while k < 3:
                v = (image[column+i, row+j, k]*mean_kernel[i][j])
                temp2.append(v)
                k = k + 1
            j = j + 1
            temp.append(temp2)
            temp2 = []
        i = i + 1
    fin = []
    for num in temp:
        fin.append(num)
    sumblue = (fin[0][0] + fin[1][0] + fin[2][0] + fin[3][0] + fin[4][0] + fin[5][0] + fin[6][0] + fin[7][0] + fin[8][0])
    sumgreen = (fin[0][1] + fin[1][1] + fin[2][1] + fin[3][1] + fin[4][1] + fin[5][1] + fin[6][1] + fin[7][1] + fin[8][1])
    sumred = (fin[0][2] + fin[1][2] + fin[2][2] + fin[3][2] + fin[4][2] + fin[5][2] + fin[6][2] + fin[7][2] + fin[8][2])
    final = [sumblue/data3, sumgreen/data3, sumred/data3]
    output_image2[column, row] = final


def rgb_to_greyscale(column, row):
    wr = 0
    wg = 1
    wb = 0
    i = (wr*image[column, row, 2] + wg*image[column, row, 1] + wb*image[column, row, 0])
    greyscaled_image[column, row] = i


image = cv2.imread("Grass1.jpg")

mean_kernel = [[1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1]]

data1 = np.shape(image)
data2 = np.shape(mean_kernel)
data3 = data2[0]*data2[1]

kernal_radius = math.floor((len(mean_kernel)-1)/2)
hor = (len(image) - 2 * kernal_radius)
ver = (len(image[0]) - 2 * kernal_radius)

greyscaled_image = np.zeros((data1[0], data1[1]), dtype=np.uint8)
output_image = np.zeros((hor, ver), dtype=np.uint8)
output_image2 = np.zeros((hor, ver, 3), dtype=np.uint8)

i = 0
while i < len(image):
    j = 0
    inner_array = image[i]
    while j < len(inner_array):
        rgb_to_greyscale(i, j)
        j = j + 1
    i = i + 1

# i = 0
# while i < len(output_image):
#     j = 0
#     inner_array = output_image[j]
#     while j < len(inner_array):
#        apply_filter_grey(i, j)
#         j = j + 1
#     i = i + 1

# i = 0
# while i < len(output_image2):
#     j = 0
#     inner_array = output_image2[j]
#     while j < len(inner_array):
#         apply_filter(i, j)
#         j = j + 1
#     i = i + 1

ret, thresh1 = cv2.threshold(greyscaled_image, 245, 255, cv2.THRESH_BINARY)

cv2.imshow("RGB", image)
cv2.imshow("Greyscaled", greyscaled_image)
cv2.imshow('Temp', thresh1)
# cv2.imshow("Grey_Filtered", output_image)
# cv2.imshow("RGB_Filtered", output_image2)
cv2.waitKey(0)
