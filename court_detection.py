from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2
import numpy as np
from os import path

detector = "HOG"
videoPath = "videos/NETS at LAKERS _ FULL GAME HIGHLIGHTS _ February 18, 2021-vNQ1qX8zn94_clipped.mp4"
print(path.exists(videoPath))

cap = cv2.VideoCapture(videoPath)

CLASSES = ["person"]

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Hard-Coded Color
    #court_color = np.uint8([[[188, 218, 236]]])
    court_color = np.uint8([[[189, 204, 233]]])

    hsv_court_color = cv2.cvtColor(court_color, cv2.COLOR_BGR2HSV)
    hue = hsv_court_color[0][0][0]

    # define range of blue color in HSV - Again HARD CODED! :(
    lower_color = np.array([hue-5 , 10, 10])
    upper_color = np.array([hue+5 , 225, 225])

    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Opening
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=opening)
    cv2.imshow('res', res)
    # Canny Edge Detector
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
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
        a,b,c = lines.shape
        for i in range(2):
            for x1, y1, x2, y2 in lines[i]:
                # cv2.line(image, start_point, end_point, color, thickness)
                cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 5)

    # Detect People
    if detector == "HOG":
        # initialize the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        orig = frame.copy()

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.70)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()