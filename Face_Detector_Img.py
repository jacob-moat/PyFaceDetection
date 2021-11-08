import cv2
from random import randrange

# pre trained face data from OpenCV
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('meandkaty.jpeg')

# converts photo to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detects the faces (rectangle)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# drawing rectangle around face dynamically in a loop
for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x+w, y+h),  (randrange(256), randrange(256), randrange(256)), 4)

# image viewer
cv2.imshow('Face Detector', img)

# make program wait until key is pressed
cv2.waitKey()

