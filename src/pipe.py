import json
import sys
from unittest import result
import cv2
import mediapipe as mp
import numpy as np

#video
video_file = 'ropejump_cut.mp4'

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error opening video stream or file.")
    raise TypeError
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    pose_landmarks = results.pose_world_landmarks
    
