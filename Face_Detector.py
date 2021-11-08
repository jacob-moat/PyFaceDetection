import cv2
from random import randrange

# pre trained face data from OpenCV
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture video
webcam = cv2.VideoCapture(0)
# 0 is default video device on pc, change 0 to video file to see facial recognition on video

# makes detection run on all frames
while True:
        successful_frame_read, frame = webcam.read()

        # converts photo to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detects the faces (rectangle)
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

        # drawing rectangle around face dynamically in a loop
        for (x, y, w, h) in face_coordinates:
                cv2.rectangle(frame, (x, y), (x+w, y+h),  (randrange(256), randrange(256), randrange(256)), 3)

        # Opens video viewer
        cv2.imshow('Face Detector', frame)

        key = cv2.waitKey(1)

        # assigns the Q or q key to close program
        if key == 81 or key == 113:
                break
                webcam.release()