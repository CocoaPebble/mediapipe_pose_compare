import json
import math

import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt

selected_jts = [
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]


def angle_2p_3d(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])

    v1mag = np.sqrt([v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]])
    v1norm = np.array([v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

    v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    v2norm = np.array([v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
    angle_rad = np.arccos(res)

    return round(math.degrees(angle_rad), 2)


def get_mediapipe_joint_angles(kpts):
    left_shoulder_angle = angle_2p_3d(kpts[selected_jts.index('left_elbow')],
                                      kpts[selected_jts.index('left_shoulder')],
                                      kpts[selected_jts.index('left_hip')])
    left_elbow_angle = angle_2p_3d(kpts[selected_jts.index('left_wrist')],
                                   kpts[selected_jts.index('left_elbow')],
                                   kpts[selected_jts.index('left_shoulder')])
    left_hip_angle = angle_2p_3d(kpts[selected_jts.index('right_hip')],
                                 kpts[selected_jts.index('left_hip')],
                                 kpts[selected_jts.index('left_knee')])
    left_knee_angle = angle_2p_3d(kpts[selected_jts.index('left_hip')],
                                  kpts[selected_jts.index('left_knee')],
                                  kpts[selected_jts.index('left_ankle')])

    right_shoulder_angle = angle_2p_3d(kpts[selected_jts.index('right_elbow')],
                                       kpts[selected_jts.index('right_shoulder')],
                                       kpts[selected_jts.index('right_hip')])
    right_elbow_angle = angle_2p_3d(kpts[selected_jts.index('right_wrist')],
                                    kpts[selected_jts.index('right_elbow')],
                                    kpts[selected_jts.index('right_shoulder')])
    right_hip_angle = angle_2p_3d(kpts[selected_jts.index('left_hip')],
                                  kpts[selected_jts.index('right_hip')],
                                  kpts[selected_jts.index('right_knee')])
    right_knee_angle = angle_2p_3d(kpts[selected_jts.index('right_hip')],
                                   kpts[selected_jts.index('right_knee')],
                                   kpts[selected_jts.index('right_ankle')])

    return [left_shoulder_angle, left_elbow_angle, left_hip_angle, left_knee_angle,
            right_shoulder_angle, right_elbow_angle, right_hip_angle, right_knee_angle]


def read_gt_pts(gt_file):
    with open(gt_file, 'r') as f:
        data = json.load(f)

    all_arr = []
    for frame in data:
        arr = []
        for joint in selected_jts:
            arr.append(data[frame][joint])
        all_arr.append(arr)
    all_arr = np.array(all_arr)
    return all_arr


def angle_error(gt, mp):
    err = np.abs(np.array(gt) - np.array(mp))
    # print(err)
    print(round(np.sum(err), 2))
    return round(np.sum(err), 2)


def draw_line(errors):
    x = range(0, 100)
    plt.plot(x, errors)
    plt.xlabel("frames")
    plt.ylabel('abs error')
    plt.show()


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

video_file = r'data\yoga4.mp4'
# gt_file = r'..\data\frontyoga100.json'
# gt_pts = read_gt_pts(gt_file)

# print(gt_pts_0)
cap = cv2.VideoCapture(video_file)
kpts_3d = []
frame_num = 0
all_errors = []

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    frame_kps = []
    if results.pose_world_landmarks:
        for i, landmark in enumerate(results.pose_world_landmarks.landmark):
            if i not in pose_keypoints:
                continue
            kpts = [landmark.x, landmark.y, landmark.z]
            frame_kps.append(kpts)
    else:
        frame_kps = [-1, -1, -1] * len(pose_keypoints)

    angles = get_mediapipe_joint_angles(frame_kps)
    print(frame_num, angles)
    gt_pts_angle = get_mediapipe_joint_angles(gt_pts[frame_num])
    print(gt_pts_angle)
    err = angle_error(gt_pts_angle, angles)
    all_errors.append(err)
    kpts_3d.append(frame_kps)

    frame_num += 1
    # if frame_num == 10:
    #     break

# kpts_3d = np.array(kpts_3d)
# draw_line(all_errors)
