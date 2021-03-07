
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

def createMask(cap):
	ret, frame = cap.read()
	capWidth = int(cap.get(3))
	capHeight = int(cap.get(4))
	mask = np.zeros([capHeight,capWidth], dtype=np.uint8)
	print("Mask Shape: ", np.array(mask).shape)
	return mask

def stabilization(cx, cy, frame):
	x = 725
	y = 485
	return x-cx, y-cy

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)
back = cv2.imread("mask.jpg")
back = cv2.resize(back, (1450,970))
back_copy = back.copy()
# mask = createMask(cap)

while True:
	ret, frame = cap.read()
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

	back[y:y+480, x:x+640] = frame[0:480, 0:640]
	# cv2.rectangle(back, (405, 245), (405 + 640, 245 + 480), color, 2)
	cv2.imshow('back', frame)
	back[y:y+480, x:x+640] = back_copy[y:y+480, x:x+640]




	# cv2.imshow('frame', frame)
	# back[200:680, 300:940] = back_copy[200:680, 300:940]
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

# mask = cv2.imread("mask.jpg")
# mask = cv2.resize(mask, (1080, 720))
# person = cv2.imread("person.jpg")
#
# cv2.imshow("mask", mask)
# cv2.imshow("person", person)
# cv2.waitKey()
