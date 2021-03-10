import numpy as np
import dlib
import cv2

def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


# pre-trained model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
# prototxt has the information of where the training data is located.
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detectFacesWithDNN(frame):
    # A neural network that really supports the input value
    size = (300,300)

    # After executing the average reduction, the image needs to be scaled
    scalefactor = 1.0

    # These are our mean subtraction values. They can be a 3-tuple of the RGB means or
    # they can be a single value in which case the supplied value is subtracted from every
    # channel of the image.
    swapRB = (104.0, 117.0, 123.0)

    height, width = frame.shape[:2]
    resizedFrame = cv2.resize(frame, size)
    blob = cv2.dnn.blobFromImage(resizedFrame, scalefactor, size, swapRB)
    print("blob: ", blob)
    net.setInput(blob)
    dnnFaces = net.forward()
    for i in range(dnnFaces.shape[2]):
        confidence = dnnFaces[0, 0, i, 2]
        if confidence > 0.5:
            box = dnnFaces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x1, y1), (193, 69, 42), 2)
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detectFacesWithDNN(frame)

    cv2.imshow("Dlib Test", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
