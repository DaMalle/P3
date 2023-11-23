import cv2
import numpy as np
import os


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

        # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)


""""" 
# Code for running through a video and saving every frame to a directory!
video = cv2.VideoCapture('hq_camera_test_17.h264')
i = 0
while True:
    try:
        ret, frame = video.read()
        cv2.imshow('video', frame)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join('C:\\Users\\kkl\\PycharmProjects\\Lecture4\\Frames', 'Frame' + str(i) + '.png'), frame)
        i += 1
    except:
        cv2.destroyAllWindows()
        break
"""""

# driver function
if __name__=="__main__":

    # Code for calibrating, so we have a rotation matrix that gets a top-down view.
    img = cv2.imread('41,5x41,5.png')
    print((img.shape[1], img.shape[0]))

    # Coordinates in (x,y) / (width, height)
    pt_A = [478, 396]
    pt_B = [802, 394]
    pt_C = [428, 719]
    pt_D = [829, 715]
    scr = np.float32([pt_A, pt_B, pt_C, pt_D])

    cv2.line(img, (pt_A[0], pt_A[1]), (pt_B[0], pt_B[1]), (0, 255, 0), 2)
    cv2.line(img, (pt_C[0], pt_C[1]), (pt_D[0], pt_D[1]), (0, 255, 0), 2)
    cv2.line(img, (pt_A[0], pt_A[1]), (pt_C[0], pt_C[1]), (0, 255, 0), 2)
    cv2.line(img, (pt_B[0], pt_B[1]), (pt_D[0], pt_D[1]), (0, 255, 0), 2)

    AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxWidth = max(int(AB), int(CD))

    AC = np.sqrt(((pt_A[0] - pt_C[0]) ** 2) + ((pt_A[1] - pt_C[1]) ** 2))
    BD = np.sqrt(((pt_B[0] - pt_D[0]) ** 2) + ((pt_B[1] - pt_D[1]) ** 2))
    maxHeight = max(int(AC), int(BD))

    print(maxWidth, maxHeight)
    max_total = max(maxWidth, maxHeight)
    print(max_total)

    center = [int(img.shape[1]/2), int(img.shape[0]/2)]  # Center coordinates in (width, height)
    print(center)

    dst = np.float32([[640 - max_total / 4, (img.shape[0] - max_total) / 2 + 360],
                      [640 + max_total / 4, (img.shape[0] - max_total) / 2 + 360],
                      [640 - max_total / 4, (img.shape[0]) / 2 + 360],
                      [640 + max_total / 4, (img.shape[0]) / 2 + 360]])

    M = cv2.getPerspectiveTransform(scr, dst)
    np.savez('BirdRotArdu', BirdRot=M)
    out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    print((out.shape[1], out.shape[0]))
    cv2.imshow('hello', img)
    cv2.imshow('Bird', out)

    cv2.setMouseCallback('hello', click_event)

    cv2.waitKey()

    # We now apply the rotation matrix to a video!
    t_frames = len(os.listdir('./Frames'))
    i = 0
    pictures = []
    video = cv2.VideoCapture('hq_camera_test_17.h264')
    fps = video.get(cv2.CAP_PROP_FPS)

    while i < t_frames:
        picture = os.listdir('./Frames')[i]
        img = ('Frames/' + str(picture))
        curr_img = cv2.imread(img)
        out = cv2.warpPerspective(curr_img, M, (curr_img.shape[1], curr_img.shape[0]))
        cv2.imshow('Image', out)
        cv2.waitKey(10)
        i += 1

