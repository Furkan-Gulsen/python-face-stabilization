
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
	resX = x - cx
	resY = y - cy
	if cx > 470:
		resX = 180
	elif cx < 195:
		resX = 450
	if cy > 325:
		resY = 120
	elif cy < 140:
		resY = 300
	return resX, resY

def detectFacesWithDNN(frame):
	size = (300,300)
	scalefactor = 1.0
	swapRB = (104.0, 117.0, 123.0)

	height, width = frame.shape[:2]
	resizedFrame = cv2.resize(frame, size)
	blob = cv2.dnn.blobFromImage(resizedFrame, scalefactor, size, swapRB)
	net.setInput(blob)
	dnnFaces = net.forward()
	for i in range(dnnFaces.shape[2]):
		confidence = dnnFaces[0, 0, i, 2]
		if confidence > 0.5:
			box = dnnFaces[0, 0, i, 3:7] * np.array([width, height, width, height])
			(x, y, x1, y1) = box.astype("int")
			# cv2.rectangle(frame, (x, y), (x1, y1), (193, 69, 42), 2)
	return frame, x, y, x1, y1

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

# pre-trained model
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
# prototxt has the information of where the training data is located.
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

cap = cv2.VideoCapture(0)
background, backgroundCopy = prepareBackground(cap)

while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame, (720,480))

	if not ret:
		break

	try:
		frame, x, y, xw, yh = detectFacesWithDNN(frame)
		cy = int((y+yh)/2)
		cx = int((x+xw)/2)
		color = (255,255,255)
		# cv2.circle(frame, (cx, cy), 4, color, 2)
		x,y = stabilization(cx,cy,frame)
		# cv2.moveWindow('Face Stabilization with DNN',int(x),int(y))
	except:
		print("There is no face")


	background[y:y+480, x:x+720] = frame[0:480, 0:720]
	cv2.rectangle(background, (450, 300), (900, 600), color, 2)
	# cv2.imshow('Bacground Process', background)
	cv2.imshow('Face stabilization with DLIB', background[300:600,450:900])
	background[y:y+480, x:x+720] = backgroundCopy[y:y+480, x:x+720]

	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
