import numpy as np
import cv2 as cv
import requests
import pyrtmp
rtsp_url = "rtmp://media.tta-edu.com/live/1581F6Q8D23CM00A866F_81-0-0"

cap = cv.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
