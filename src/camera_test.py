import sys

import cv2
import mediapipe as mp
import numpy as np

from calculate_joint_angle import get_mediapipe_joint_angles

# gt_file = r'data\frontyoga100.json'
key_pose_image = 'data\img\keypose.png'
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
angle_order = ['left_shoulder_angle', 'left_elbow_angle',
               'left_hip_angle', 'left_knee_angle',
               'right_shoulder_angle', 'right_elbow_angle',
               'right_hip_angle', 'right_knee_angle']


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r'C:\Users\jyz18\Documents\realsense\rgbd_test.mp4')
if not cap.isOpened():
    print("Error opening video stream or file.")
    raise TypeError
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

img = cv2.imread(key_pose_image)
if img is None:
    sys.exit("no image file")
img = ResizeWithAspectRatio(img, width=frame_width, height=frame_height)
img = cv2.flip(img, 1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# gt_pose = read_gt_pts(gt_file)
# gt_angle_90 = [110.06, 172.89, 108.2, 161.27, 94.07, 159.08, 94.36, 163.82]
gt_angle_90 = [109.0, 171.31, 105.29, 163.41, 95.44, 161.23, 91.65, 167.87]
all_error = []
frame_num = 0

while True:
    ret, image = cap.read()
    if not ret:
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    frame_kps = []
    err = 0
    err_arr = []
    pred_angles = []
    if results.pose_world_landmarks:
        for i, landmark in enumerate(results.pose_world_landmarks.landmark):
            if i not in pose_keypoints:
                continue
            kpts = [landmark.x, landmark.y, landmark.z]
            frame_kps.append(kpts)

        pred_angles = get_mediapipe_joint_angles(frame_kps)
        print(frame_num, pred_angles)
        err_arr = np.abs(np.array(gt_angle_90) - np.array(pred_angles))
        err = np.round(np.sum(err_arr), 3)
        print(err, str(err), err_arr)
    else:
        print('not detected')
        err_arr = [0, 0, 0, 0, 0, 0, 0, 0]
        frame_kps = [-1, -1, -1] * len(pose_keypoints)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    frame_num += 1

    image = cv2.flip(image, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    thickness = 3
    linetype = 3

    cv2.putText(image, str(err), (20, 100), font, font_scale, font_color, thickness=thickness, lineType=linetype)
    ypos = 140
    for i, ele in enumerate(err_arr):
        # text = angle_order[i] + ' ' + str(round(ele, 2))
        text = angle_order[i] + ' ' + str(round(pred_angles[i]))
        cv2.putText(image, text, (20, ypos + i * 40), font, font_scale, font_color, thickness=thickness,
                    lineType=linetype)
    numpy_horizontal = np.hstack((img, image))
    cv2.imshow('Pose detection', numpy_horizontal)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
