#imports
import numpy as np
import cv2
import statistics as s
import math
import matplotlib.pyplot as plt


video_capture = cv2.VideoCapture(0)
cv2.namedWindow("Window")

while True:
	ret, frame = video_capture.read()
	print("reading")
    #Mostrar webcam
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()