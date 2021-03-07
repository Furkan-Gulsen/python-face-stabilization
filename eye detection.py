
# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dlib
import cv2

# Returns the x, y coordinates of 68 points taken on the
# face as a 2-tuple-shaped list.
def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def drawEye(thresh, mid, frame, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

faceLandmarks = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 1)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        shape = shapePoints(shape)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mid = (shape[42][0] + shape[39][0]) // 2


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows();
