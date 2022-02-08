import json
import numpy as np


joint_list = [
    'mid_hip',
    'right_hip', 'right_knee', 'right_ankle', 'right_foot_index',
    'left_hip', 'left_knee', 'left_ankle', 'left_foot_index',
    'spine',
    'mid_shoulder',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist'
]

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

temp_joint_list = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']

temp_dict = {
    'left_shoulder': [],
    'right_shoulder': [],
    'left_hip': [],
    'right_hip': []
}

mp_dict = {}
frame_idx = 1
frame_num = 0
flag = 0

filepath = "ropejump2_world_landmark_result.json"
with open(filepath, 'r') as f:
    data = json.load(f)

for ele in data:
    # Check whether the 1st frame is Frame 1
    if flag == 0:
        flag = 1
        frame_idx = ele["frame"]
        mp_dict[frame_idx] = {}
        # print(mp_dict[frame_idx])

    # If frame_idx differs from the previous run, add middle_hip, middle_shoulder and spine joints
    # if ele["frame"] != frame_idx:

    if ele["frame"] != frame_idx:
        frame_idx = ele["frame"]
        mp_dict[frame_idx] = {}
        # print(mp_dict[frame_idx])

    # Add data to dictionary if no change in ele["frame"]
    joint_name = landmark_names[(ele["keypoint_num"])]

    # print(mp_dict[frame_idx])
    mp_dict[frame_idx][joint_name] = [ele["X"], ele["Y"], ele["Z"]]
    # print(mp_dict)
    # print(mp_dict[frame_idx])

    if len(mp_dict[frame_idx]) == len(landmark_names):
        mp_dict[frame_idx]['mid_hip'] = [0, 0, 0]
        mp_dict[frame_idx]['mid_shoulder'] = [0, 0, 0]
        mp_dict[frame_idx]['spine'] = [0, 0, 0]

        for i in range(3):
            mp_dict[frame_idx]['mid_hip'][i] = (
                mp_dict[frame_idx]['left_hip'][i] + mp_dict[frame_idx]['right_hip'][i]) * 0.5
            mp_dict[frame_idx]['mid_shoulder'][i] = (
                mp_dict[frame_idx]['left_shoulder'][i] + mp_dict[frame_idx]['right_shoulder'][i]) * 0.5
            mp_dict[frame_idx]['spine'][i] = (
                mp_dict[frame_idx]['mid_shoulder'][i] - mp_dict[frame_idx]['mid_hip'][i]) * 0.25

        frame_num += 1

with open('ropejump2.json', 'w') as f:
    json.dump(mp_dict, f, indent=4)

first_key = list(mp_dict.keys())[0]
full_array = np.zeros((frame_num, len(joint_list), 3))


for index in range(frame_num):
    for i in range(len(joint_list)):
        loc = np.array([mp_dict[index+int(first_key)][joint_list[i]]])
        full_array[index, i, :] = loc

np.save("ropejump2", full_array)
