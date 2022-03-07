import json
import math
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from bvh import Bvh
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

bvh_joint_list = [
    'mid_hip',
    'left_hip', 'left_knee',
    'left_ankle', 'left_foot_index',
    'spine',
    'mid_shoulder', 'left_shoulder',
    'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'right_hip', 'right_knee',
    'right_ankle', 'right_foot_index',
]


def get_center(a, b):
    """Calculates pose center as point between hips."""
    left_hip = np.array(a)
    right_hip = np.array(b)
    center = (left_hip + right_hip) * 0.5
    return center


def get_spine(a, b):
    mid_shoulder = np.array(a)
    mid_hip = np.array(b)
    center = (mid_shoulder - mid_hip) * 0.25
    return center


def get_all_standard_pose(file):
    # bvh frame start from 0
    all_channels = []
    channels = ['Zrotation', 'Xrotation', 'Yrotation']

    with open(file) as f:
        mocap = Bvh(f.read())
    total_frames = mocap.nframes

    for i in range(total_frames):
        joint_channels = []
        for joint in bvh_joint_list:
            joint_channels.append(
                mocap.frame_joint_channels(i-1, joint, channels))
        all_channels.append(joint_channels)

    return np.array(all_channels, dtype=np.float32)


def get_key_pose(file, frame):
    # bvh frame start from 0

    joint_channels = []
    channels = ['Zrotation', 'Xrotation', 'Yrotation']

    with open(file) as f:
        mocap = Bvh(f.read())

    for joint in bvh_joint_list:
        joint_channels.append(
            mocap.frame_joint_channels(frame-1, joint, channels))

    return np.array(joint_channels, dtype=np.float32)


def calculate_sse(sa, aa):
    # standard angles [], actual angles []
    sa_rad = np.radians(sa.ravel())
    aa_rad = np.radians(aa.ravel())
    err = np.cos(sa_rad) - np.cos(aa_rad)
    return np.dot(err, err)


def draw_line(frames, errors, start, end):
    x = range(start, end)
    plt.plot(x, errors[start:end])
    plt.title(frames)
    plt.xlabel("frames")
    plt.ylabel('square sum error')
    plt.show()


def draw_17_joint_line(frames, errors, start, end, joint):
    x = range(start, end)
    color_list = ['blue', 'chocolate', 'crimson', 'darkcyan',
                  'darkred', 'green', 'red', 'pink',
                  'purple', 'grey', 'brown', 'olive',
                  'cyan', 'black', 'mediumturquoise', 'navy', 'orange']
    for i, ele in enumerate(joint_list):
        if i == joint:
            plt.plot(x, errors[i][start:end], color=color_list[i], label=ele)
    plt.title(frames)
    plt.legend(loc="best")
    plt.xlabel('frames')
    plt.ylabel('errors')
    plt.show()


def write_predict_bvh(results, out):
    npa = np.asarray(results, dtype=np.float32)
    mpsk = MediaPipeSkeleton()
    channel, header = mpsk.poses2bvh(npa, output_file=out)
    print('write predicted bvh file,', out)


def save_mp_result():
    ...


def each_joint_calculate_sse(all_std_pose, all_pred_pose, key_pose_frame, frame_num):
    period = 20
    rows = []
    for frame in range(1, frame_num):
        cur_row = [frame]
        for i, joint in enumerate(bvh_joint_list):
            # print('frame', i, joint, all_pose[frame][i])
            # print('key_pose', joint, all_pose[key_pose_frame][i])
            # print(i, joint, calculate_sse(all_pose[frame][i], all_pose[key_pose_frame][i]))
            cur_row.append(
                round(calculate_sse(all_pred_pose[frame][i], all_std_pose[frame][i]), 3))
        rows.append(cur_row)
    print(rows)


def get_header(std_npy_file):
    data = np.load(std_npy_file)
    mpsk = MediaPipeSkeleton()
    return mpsk.get_bvh_header(data)


def main():
    with open("src\config.json", "r") as f:
        config = json.load(f)

    # start camera or import video
    cap = cv2.VideoCapture(config['video_file'])
    if not cap.isOpened():
        print("Error opening video stream or file.")
        raise TypeError
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_num = 0

    # import MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mpsk = MediaPipeSkeleton()

    # get one standard pose at frame
    std_pose = get_key_pose(config['std_bvh_file'], config['key_pose_frame'])

    # get standard t-pose skeleton
    std_header = None
    if config['header_file']:
        std_header = get_header(config['header_file'])

    # get all standard pose
    all_pose = get_all_standard_pose(config['std_bvh_file'])

    # calculate gt each joint error
    # each_joint_calculate_sse(all_pose, key_pose_frame)

    # initialize prediction array
    all_array = []
    all_error_sum = []
    test = []

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
            # print(frame_num, "not detected")
            all_error_sum.append(0)  # add no error if not predicted
            continue

        for kp_num, data_point in enumerate(world_landmarks.landmark):
            keypoints.append({
                'frame': frame_num,
                'keypoint_num': kp_num,
                'X': data_point.x,
                'Y': data_point.y,
                'Z': data_point.z,
            })

        joint_group = {}  # points in each frame

        # select specific points for bvh
        for idx, ele in enumerate(keypoints):
            joint_name = landmark_names[(ele["keypoint_num"])]
            if joint_name in joint_list:
                joint_group[joint_name] = [ele["X"], ele["Y"], ele["Z"]]

        joint_group['mid_hip'] = get_center(
            joint_group['left_hip'], joint_group['right_hip']).tolist()
        joint_group['mid_shoulder'] = get_center(
            joint_group['left_shoulder'], joint_group['right_shoulder']).tolist()
        joint_group['spine'] = get_spine(
            joint_group['mid_shoulder'], joint_group['mid_hip']).tolist()

        # convert to array
        arr = []
        for i, joint_name in enumerate(joint_list):
            arr.append(joint_group[joint_name])

        # process predict results
        npa = np.array([arr], dtype=np.float32)
        channel, header = mpsk.poses2bvh(npa, header=std_header)
        channel = np.array(channel[0][3:])  # remove first 3 position, (1, 51)
        actual_channels = np.reshape(np.ravel(channel), (17, 3))

        # test_cur = [frame_num]
        # for i, joint in enumerate(bvh_joint_list):
        #     e = calculate_sse(std_pose[i], actual_channels[i])
        #     test_cur.append(e)
        #     print(i, joint, e)
        # test.append(test_cur)

        error_sum = calculate_sse(std_pose, actual_channels)
        all_error_sum.append(error_sum)
        print(frame_num, error_sum)
        all_array.append(arr)

    pose.close()

    print('#'*80)
    print('predict video:', config['video_file'])
    print('standard bvh:', config['std_bvh_file'])
    print('#'*80)

    # print(test)
    # draw_line(config['key_pose_frame'], all_error_sum, 1, frame_num)
    # np_all_array = np.array(all_array)
    # each_joint_calculate_sse(all_pose, np_all_array, config['key_pose_frame'], frame_num)
    # write_predict_bvh(all_array, output_file)


if __name__ == '__main__':
    main()
