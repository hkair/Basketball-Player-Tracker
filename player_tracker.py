from __future__ import print_function
from imutils.object_detection import non_max_suppression
import cv2
import numpy as np
from easydict import EasyDict
from random import randint
import sys
from imutils.video import FPS

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]
print(cv2.__version__)

args = EasyDict({

    'detector': "tracker",

    # Path Params
    'videoPath': "videos/NETS at LAKERS _ FULL GAME HIGHLIGHTS _ February 18, 2021-vNQ1qX8zn94_clipped.mp4",

    # Player Tracking
    'classes': ["person"],
    'tracker': "CSRT",
    'trackerTypes': ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'],
    'singleTracker': False,

    # Court Line Detection
    'draw_line': False,

    # YOLOV3 Detector
    'weights': "yolov3.weights",
    'config': "yolov3.cfg",

})

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == args.trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == args.trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == args.trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == args.trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == args.trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == args.trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == args.trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == args.trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in args.trackerTypes:
            print(t)

    return tracker

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    # Indices 0 is for person
    if class_id == 0:
        label = str(args.classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        # Text of Class
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        if args.singleTracker:
            tracker = cv2.Tracker_create(args.tracker.upper())

    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # appropriate object tracker constructor:
    else:
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "mil": cv2.TrackerMIL_create
        }

        if args.singleTracker:
            tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()

    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None
    # initialize the FPS throughput estimator
    fps = None

    COLORS = np.random.uniform(0, 255, size=(len(args.classes), 3))
    # Set up Neural Net
    net = cv2.dnn.readNet(args.weights, args.config)

    cap = cv2.VideoCapture(args.videoPath)

    player_threshold = 99999

    if not args.singleTracker:
        # Read first frame
        success, frame = cap.read()
        # quit if unable to read the video file
        if not success:
            print('Failed to read video')
            sys.exit(1)

        ## Select boxes
        bboxes = []
        colors = []

        # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
        # So we will call this function in a loop till we are done selecting all objects
        while True:
            # draw bounding boxes over objects
            # selectROI's default behaviour is to draw box starting from the center
            # when fromCenter is set to false, you can draw box starting from top left corner
            bbox = cv2.selectROI('MultiTracker', frame)
            bboxes.append(bbox)
            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            print(k)
            if (k == 113):  # q is pressed
                break

        print('Selected bounding boxes {}'.format(bboxes))

        createTrackerByName(args.tracker)

        # Create MultiTracker object
        trackers = cv2.legacy.MultiTracker_create()

        # Initialize MultiTracker
        for bbox in bboxes:
            trackers.add(createTrackerByName(args.tracker), frame, bbox)

    while(cap.isOpened()):

        # Take each frame
        _, frame = cap.read()

        Width = frame.shape[1]
        Height = frame.shape[0]

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

        if args.draw_line:
            # Canny Edge Detector
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('gray', gray)

            high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            low_thresh = 0.5 * high_thresh
            edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=3)
            cv2.imshow('Canny Edge Detector', edges)

            # # Hough Lines
            minLineLength = 200
            maxLineGap = 500
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

            # Green color in BGR
            LINE_COLOR = (255, 0, 0)

            if lines is None:
                continue
            else:
                a,b,c = lines.shape
                for i in range(2):
                    for x1, y1, x2, y2 in lines[i]:
                        # cv2.line(image, start_point, end_point, color, thickness)
                        if args.draw_line:
                            cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 3)
                        # only compare the lower corner of y value
                        player_threshold = min(player_threshold, y1, y2)

        # Detect People
        if args.detector == "HOG":
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
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.1)
            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        elif args.detector == "yolov3":
            scale = 0.00392
            blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            k = 0
            for i in indices:
                i = i[0]
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                pad = 5
                #print(player_threshold)
                if (round(y+h) < player_threshold):
                    k+=1
                    continue
                else:
                    draw_prediction(frame, class_ids[i], round(x-pad), round(y-pad), round(x + w + pad), round(y + h + pad))

        elif args.detector == "tracker":

            # check to see if we are currently tracking an object
            if args.singleTracker:
                if initBB is not None:
                    # grab the new bounding box coordinates of the object
                    (success, box) = tracker.update(frame)
                    # check to see if the tracking was a success
                    if success:
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 255, 0), 2)
                    # update the FPS counter
                    fps.update()
                    fps.stop()
                    # initialize the set of information we'll be displaying on
                    # the frame
                    info = [
                        ("Tracker", tracker),
                        ("Success", "Yes" if success else "No"),
                        ("FPS", "{:.2f}".format(fps.fps())),
                    ]
                    # loop over the info tuples and draw them on our frame
                    for (i, (k, v)) in enumerate(info):
                        text = "{}: {}".format(k, v)
                        cv2.putText(frame, text, (10, Height - ((i * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # get updated location of objects in subsequent frames
                success, boxes = trackers.update(frame)

                # draw tracked objects
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        else:
            continue

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            if args.singleTracker:
                # select the bounding box of the object we want to track (make
                # sure you press ENTER or SPACE after selecting the ROI)
                initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                                       showCrosshair=True)
                # start OpenCV object tracker using the supplied bounding box
                # coordinates, then start the FPS throughput estimator as well
                tracker.init(frame, initBB)
                fps = FPS().start()
                # if the `q` key was pressed, break from the loop

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()