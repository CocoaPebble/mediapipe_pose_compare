import json
import sys
from unittest import result
import cv2
import mediapipe as mp
import numpy as np

landmark_names = [
    'nose', 
    'left_eye_inner', 'left_eye', 'left_eye_outer', 
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 
    'mouth_left', 'mouth_right', 
    'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 
    'left_pinky_1', 'right_pinky_1', 
    'left_index_1', 'right_index_1', 
    'left_thumb_2', 'right_thumb_2',
    'left_hip', 'right_hip', 
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 
    'left_heel', 'right_heel', 
    'left_foot_index', 'right_foot_index']

joint_list = [
    'mid_hip',
    'right_hip', 'right_knee', 'right_ankle', 'right_foot_index',
    'left_hip', 'left_knee', 'left_ankle', 'left_foot_index',
    'spine',
    'mid_shoulder',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist'
]

# video
video_file = 'ropejump_cut.mp4'

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error opening video stream or file.")
    raise TypeError
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_num = 0

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break
    
    frame_num += 1
    keypoints = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image)
    world_landmarks = results.pose_world_landmarks

    for kp_num, data_point in enumerate(world_landmarks.landmark):
        keypoints.append({
            'frame': frame_num,
            'keypoint_num': kp_num,
            'X': data_point.x,
            'Y': data_point.y,
            'Z': data_point.z,
        })
    
    mp_dict = {}
    for idx, ele in enumerate(keypoints):
        if ele["frame"] != idx:
            idx = ele["frame"]
            mp_dict[idx] = {}
        
        joint_name = landmark_names[(ele["keypoint_num"])]
        print(idx, joint_name, ele)
        mp_dict[idx][joint_name] = [ele["X"], ele["Y"], ele["Z"]]

        if len(mp_dict[idx]) == len(landmark_names):
            mp_dict[idx]['mid_hip'] = [0, 0, 0]
            mp_dict[idx]['mid_shoulder'] = [0, 0, 0]
            mp_dict[idx]['spine'] = [0, 0, 0]

            for i in range(3):
                mp_dict[idx]['mid_hip'][i] = (
                    mp_dict[idx]['left_hip'][i] + mp_dict[idx]['right_hip'][i]) * 0.5
                mp_dict[idx]['mid_shoulder'][i] = (
                    mp_dict[idx]['left_shoulder'][i] + mp_dict[idx]['right_shoulder'][i]) * 0.5
                mp_dict[idx]['spine'][i] = (
                    mp_dict[idx]['mid_shoulder'][i] - mp_dict[idx]['mid_hip'][i]) * 0.25
    
    first_key = list(mp_dict.keys())[0]
    full_array = np.zeros((len(joint_list), 3))

    for i in range(len(joint_list)):
        loc = np.array([mp_dict[index+int(first_key)][joint_list[i]]])
        full_array[index, i, :] = loc
    
    print(full_array)