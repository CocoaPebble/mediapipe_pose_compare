import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video stream or file.")
    raise TypeError
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

image = cv2.imread(r'data\1.jpg')
image = cv2.resize(image, (frame_width, frame_height))
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

cv2.imshow('Numpy Horizontal', numpy_horizontal)

cv2.waitKey()