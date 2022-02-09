import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe_skeleton import MediaPipeSkeleton

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


def get_center(a, b):
    """Calculates pose center as point between hips."""
    left_hip = np.array(a)
    right_hip = np.array(b)
    center = (left_hip + right_hip) * 0.5
    return center


def get_spine(a, b):
    mid_hip = np.array(a)
    mid_shoulder = np.array(b)
    center = (mid_shoulder - mid_hip) * 0.25
    return center


# video
video_file = 'ropejump.mp4'

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
mp_dict = {}
poses = []
full_array = []
time2total = 0
time3total = 0
time4total = 0


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
    if world_landmarks is None:
        print(frame_num, "is none")
        continue

    for kp_num, data_point in enumerate(world_landmarks.landmark):
        keypoints.append({
            'frame': frame_num,
            'keypoint_num': kp_num,
            'X': data_point.x,
            'Y': data_point.y,
            'Z': data_point.z,
        })
        point = [data_point.x, data_point.y, data_point.z]

    frame_group = {}  # points in each frame

    # select specific points for bvh
    for idx, ele in enumerate(keypoints):
        joint_name = landmark_names[(ele["keypoint_num"])]
        if joint_name in joint_list:
            frame_group[joint_name] = [ele["X"], ele["Y"], ele["Z"]]

    frame_group['mid_hip'] = get_center(frame_group['left_hip'], frame_group['right_hip']).tolist()
    frame_group['mid_shoulder'] = get_center(frame_group['left_shoulder'], frame_group['right_shoulder']).tolist()
    frame_group['spine'] = get_spine(frame_group['mid_shoulder'], frame_group['mid_hip']).tolist()
    mp_dict[frame_num] = frame_group

    # convert to array
    arr = []
    for i, joint_name in enumerate(joint_list):
        loc = frame_group[joint_name]
        arr.append(loc)
    full_array.append(arr)

npa = np.asarray(full_array, dtype=np.float32)

mp = MediaPipeSkeleton()
channels, header = mp.poses2bvh(npa)
subarr = channels[1][3:]

print("run with ", video_file, "frame_num", frame_num)
