
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
	x = 1000
	y = 450
	return x-cx, y-cy

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


cap = cv2.VideoCapture(0)
mask = createMask(cap)

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
		# masked = cv2.bitwise_and(mask[y: y+h, x:x+w], mask[y: y+h, x:x+w], mask=frame[y: y+h, x:x+w] )
		# frame[y: y+h, x:x+w].append(mask[y: y+h, x:x+w])
		# cv2.circle(mask, (145, 200), 100, 255, -1)
		cy = int((2*y+h)/2)
		cx = int((2*x+w)/2)
		color = (255,255,255)
		cv2.circle(frame, (cx, cy), 4, (0, 0, 255), 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)



	# cv2.imshow('Face Mask', mask)
	cv2.imshow('frame', frame)
	x,y = stabilization(cx,cy,frame)
	cv2.moveWindow('frame',int(x),int(y))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
