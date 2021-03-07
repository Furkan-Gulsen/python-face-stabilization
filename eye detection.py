
# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dlib
import cv2

faceLandmarks = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

# Returns the x, y coordinates of 68 points taken on the
# face as a 2-tuple-shaped list.
def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
LEFT_RIGHT_POINTS = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        for (x, y) in points:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)



    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows();
