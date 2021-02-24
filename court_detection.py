import cv2
import numpy as np
from os import path

videoPath = "videos/NETS at LAKERS _ FULL GAME HIGHLIGHTS _ February 18, 2021-vNQ1qX8zn94.mp4"
print(path.exists(videoPath))

cap = cv2.VideoCapture(videoPath)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HARD CODED COURT COLOR :(
    court_color = np.uint8([[[153, 204, 255]]])

    hsv_court_color = cv2.cvtColor(court_color, cv2.COLOR_BGR2HSV)
    hue = hsv_court_color[0][0][0]

    # define range of blue color in HSV - Again HARD CODED! :(
    lower_color = np.array([hue - 10, 10, 10])
    upper_color = np.array([hue + 10, 200, 200])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Opening
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=opening)

    cv2.imshow('frame', frame)
    cv2.imshow('res', res)

    # Canny Edge Detector
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('Canny Edge Detector', edges)

    # # Hough Lines
    minLineLength = 100
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=250)

    # Green color in BGR
    LINE_COLOR = (255, 0, 0)

    if lines is None:
        continue
    else:
        for x1, y1, x2, y2 in lines[0]:
            # cv2.line(image, start_point, end_point, color, thickness)
            cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 10)

        for x1, y1, x2, y2 in lines[1]:
            cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 10)

        # for x1, y1, x2, y2 in lines[2]:
        #     cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 10)

    cv2.imshow('Hough', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()