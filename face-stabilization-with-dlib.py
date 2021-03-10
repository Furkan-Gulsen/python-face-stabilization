
import cv2
import dlib
import numpy as np

def shapePoints(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def stabilization(cx, cy, frame):
	x = 650
	y = 450
	return x-cx, y-cy

def prepareBackground(cap):
	# copying frame from the camera to create the background
	ret, frame = cap.read()
	background = frame.copy()
	background = cv2.resize(background, (1300,900))
	background = cv2.GaussianBlur(background, (101,101), 0)
	# copying the background to achieve refresh
	backgroundCopy = background.copy()
	backgroundCopy = cv2.GaussianBlur(backgroundCopy, (101,101), 0)
	return background, backgroundCopy


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
background, backgroundCopy = prepareBackground(cap)

while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame, (720,480))

	if not ret:
		break
	rects = detector(frame, 0)
	for rect in rects:
		shape = predictor(frame, rect)
		points = shapePoints(shape)
		(x, y, w, h) = rectPoints(rect)
		resized = frame[y: y+h, x:x+w]
		cy = int((2*y+h)/2)
		cx = int((2*x+w)/2)
		color = (255,255,255)
		cv2.circle(frame, (cx, cy), 4, color, 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
		x,y = stabilization(cx,cy,frame)

	background[y:y+480, x:x+720] = frame[0:480, 0:720]
	cv2.rectangle(background, (450, 300), (900, 600), color, 2)
	cv2.imshow('Bacground Process', background)
	# cv2.imshow('Face stabilization with DLIB', background[300:600,450:900])
	background[y:y+480, x:x+720] = backgroundCopy[y:y+480, x:x+720]

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
