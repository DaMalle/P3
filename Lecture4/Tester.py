import cv2
import numpy as np

image = cv2.imread("Grass2.jpg")


def hsv_saturation(image):  # This code provides a BLOB with grass as larger blobs than noise.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    ret, thres = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel=(np.ones((5, 5), dtype=np.uint8)))
    dilate = cv2.dilate(opening, kernel=(np.ones((5, 5), dtype=np.uint8)))
    # cv2.imshow('Saturation', s)
    # cv2.imshow('Thres', dilate)
    cv2.imshow('Dilate', dilate)


def excessive_green(image):
    b, g, r = cv2.split(image)

    output_full = 2 * (g / (r + g + b)) - (r / (r + g + b)) - (b / (b + g + r))
    b_c = np.clip(b, 1, 256)
    g_c = np.clip(g, 1, 256)
    r_c = np.clip(r, 1, 256)

    # output_half = 2 * (g_c/(r_c+g_c+b_c)) - (r_c/(r_c+g_c+b_c)) - (b_c/(b_c+g_c+r_c))
    ret, imout = cv2.threshold(output_full, 0.5, 1, cv2.THRESH_BINARY)
    dilated = cv2.dilate(imout, kernel=(np.ones((2, 2), dtype=np.uint8)))
    close = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel=(np.ones((3, 3), dtype=np.uint8)))
    # cv2.imshow('exg1', output_full)
    # cv2.imshow('exg2', output_half)
    # cv2.imshow('Pain', imout)
    # cv2.imshow('dilate', dilated)
    cv2.imshow('close', close)


def excessive_green_hsv(image):  # Not yet functional
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    output_full = 2 * (s / (v + s + h)) - (v / (v + s + h)) - (h / (h + s + v))
    b_c = np.clip(h, 1, 256)
    g_c = np.clip(s, 1, 256)
    r_c = np.clip(v, 1, 256)

    output_half = 2 * (g_c / (r_c + g_c + b_c)) - (r_c / (r_c + g_c + b_c)) - (b_c / (b_c + g_c + r_c))
    # cv2.imshow('exg1', output_full)
    # cv2.imshow('exg2', output_half)


def hsv_colour_trim(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255)) # Original values
    mask = cv2.inRange(hsv, (40, 25, 25), (80, 255, 255))  # For testing purposes
    # mask = cv2.inRange(hsv, (36, 25, 25), (156, 255, 255))

    ## Slice the green
    imask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]

    ## Save
    # cv2.imshow('Green split', green)

    gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 70, 255, 0)
    # cv2.imshow('gray', gray)
    # cv2.imshow('green', green)

    image_split_test(thresh)


def image_split_test(image):
    height, width = image.shape
    cut_height = height // 2
    quarter_width = width // 4

    section_1 = image[:cut_height, :quarter_width]
    section_2 = image[:cut_height, quarter_width:quarter_width * 2]
    section_3 = image[:cut_height, quarter_width * 2:quarter_width * 3]
    section_4 = image[:cut_height, quarter_width * 3:quarter_width * 4]

    cv2.imshow('Sec1', section_1)
    cv2.imshow('Sec2', section_2)
    cv2.imshow('Sec3', section_3)
    cv2.imshow('Sec4', section_4)

    check_green_amount(section_1, section_2, section_3, section_4)


def check_green_amount(s1, s2, s3, s4):
    #print("Section 1: " + str(cv2.countNonZero(s1)))
    if cv2.countNonZero(s1) > 1000:
        print("Big plant in section 1")
    #print("Section 2: " + str(cv2.countNonZero(s2)))
    if cv2.countNonZero(s2) > 1000:
        print("Big plant in section 2")
    #print("Section 3: " + str(cv2.countNonZero(s3)))
    if cv2.countNonZero(s3) > 1000:
        print("Big plant in section 3")
    #print("Section 4: " + str(cv2.countNonZero(s4)))
    if cv2.countNonZero(s4) > 1000:
        print("Big plant in section 4")


def start_recording():
    video = cv2.VideoCapture(0)

    while True:
        success, frame = video.read()

        # excessive_green(frame)
        hsv_colour_trim(frame)
        cv2.waitKey(1)


# hsv_saturation(image)
# excessive_green(image)
# excessive_green_hsv(image)
#hsv_colour_trim(image)
# image_split_test(image)
start_recording()
cv2.waitKey()
