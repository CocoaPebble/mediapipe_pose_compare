import json
import sys

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     enable_segmentation=False,
#     smooth_segmentation=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

cap = cv2.VideoCapture(sys.argv[1])

if not cap.isOpened():
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind('/') + 1], sys.argv[1][sys.argv[1].rfind('/') + 1:]
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_annotated.{inflext}'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (frame_width, frame_height))

frame_num = 0
write_outfile_name = f'{outdir}{inflnm}_world_landmark_result.json'
outf = open(write_outfile_name, 'w')
keypoints = []
poses = {}

keypoints_name = [
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
    'left_foot_index', 'right_foot_index',
]

while cap.isOpened():
    ret, image = cap.read()
    frame_num += 1
    if not ret:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    pose_landmarks = results.pose_world_landmarks

    keypoints_num = 0
    body = {}
    if not pose_landmarks:
        print(frame_num, "is none")
        continue
    for data_point in pose_landmarks.landmark:
        keypoints.append({
                            'frame': frame_num,
                            'keypoint_num': keypoints_num,
                            'X': data_point.x,
                            'Y': data_point.y,
                            'Z': data_point.z,
                            'Visibility': data_point.visibility,
                            })
        # body[keypoints_name[keypoints_num]] = [data_point.x, data_point.y, data_point.z]

        poses[str(keypoints_num)] = [data_point.x, data_point.y, data_point.z]
        keypoints_num += 1

    poses[str(frame_num)] = body

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    out.write(image)

outf.writelines(json.dumps(poses, indent=2))

print(frame_num)
outf.close()
pose.close()
cap.release()
out.release()
