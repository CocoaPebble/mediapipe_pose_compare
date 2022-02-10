from tracemalloc import start
import cv2
import numpy as np
import mediapipe as mp
from mediapipe_skeleton import MediaPipeSkeleton
from bvh import Bvh
import matplotlib.pyplot as plt

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


def get_all_standard_pose(file, total_frames):
    all_channels = []
    channels = ['Xrotation', 'Yrotation', 'Zrotation']

    with open(file) as f:
        mocap = Bvh(f.read())

    for i in range(total_frames):
        joint_channels = []
        for joint in joint_list:
            joint_channels.append(
                mocap.frame_joint_channels(i, joint, channels))
        all_channels.append(joint_channels)

    return np.array(all_channels, dtype=np.float32)


def get_channels_from_frame(file, frame):
    joint_channels = []
    channels = ['Xrotation', 'Yrotation', 'Zrotation']

    with open(file) as f:
        mocap = Bvh(f.read())

    for joint in joint_list:
        joint_channels.append(
            mocap.frame_joint_channels(frame, joint, channels))

    return np.array(joint_channels, dtype=np.float32)


def calculate_sse(sa, aa):
    # standard angles [], actual angles []
    err = np.cos(sa.ravel()) - np.cos(aa.ravel())
    return np.dot(err, err)


def draw_line(frames, errors, start, end):
    x = range(start, end)
    plt.plot(x, errors[start:end])
    plt.title(frames)
    plt.xlabel("frames")
    plt.ylabel('square sum error')
    plt.show()


def main():
    ##################################
    video_file = 'ropejump.mp4'
    std_bvh_file = 'amassJumpRope0012json_rot_adjusted.bvh'
    expected_pose_frame = 150
    ##################################

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
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # get one standard pose at frame 
    standard_pose = get_channels_from_frame(std_bvh_file, expected_pose_frame)

    # get all standard pose
    # all_pose = get_all_standard_pose(std_bvh_file, 191)
    # std_errors = []
    # for i, x in enumerate(all_pose):
    #     error = calculate_sse(standard_pose, x)
    #     std_errors.append(error)

    # print(std_errors)
    # draw_line(150, std_errors, 130, 170)


    ##################################
    # print('standard_joint_channels at frame', expected_pose_frame)
    # print(np.cos(standard_joint_channels), standard_joint_channels.shape)
    ##################################

    full_array = []
    all_error_sum = []

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
            all_error_sum.append(0)
            continue

        for kp_num, data_point in enumerate(world_landmarks.landmark):
            keypoints.append({
                'frame': frame_num,
                'keypoint_num': kp_num,
                'X': data_point.x,
                'Y': data_point.y,
                'Z': data_point.z,
            })

        frame_group = {}  # points in each frame

        # select specific points for bvh
        for idx, ele in enumerate(keypoints):
            joint_name = landmark_names[(ele["keypoint_num"])]
            if joint_name in joint_list:
                frame_group[joint_name] = [ele["X"], ele["Y"], ele["Z"]]

        frame_group['mid_hip'] = get_center(
            frame_group['left_hip'], frame_group['right_hip']).tolist()
        frame_group['mid_shoulder'] = get_center(
            frame_group['left_shoulder'], frame_group['right_shoulder']).tolist()
        frame_group['spine'] = get_spine(
            frame_group['mid_shoulder'], frame_group['mid_hip']).tolist()

        # convert to array
        arr = []
        for i, joint_name in enumerate(joint_list):
            loc = frame_group[joint_name]
            arr.append(loc)

        full_array = [arr]
        npa = np.asarray(full_array, dtype=np.float32)
        mpsk = MediaPipeSkeleton()
        channel, header = mpsk.poses2bvh(npa)
        channel = np.array(channel[0][3:])
        actual_channels = np.reshape(np.ravel(channel), (17, 3))

        ##################################
        # print('actual_channels at frame', frame_num)
        # print(np.cos(actual_channels), actual_channels.shape)
        ##################################

        error_sum = calculate_sse(standard_pose, standard_pose)
        all_error_sum.append(error_sum)

        print(frame_num, error_sum)
        break
        # full_array.append(arr)

    # npa = np.asarray(full_array, dtype=np.float32)
    # mp = MediaPipeSkeleton()
    # output_file = 'output.bvh'
    # channel, header = mp.poses2bvh(npa, output_file=output_file)
    draw_line(frame_num, all_error_sum, 90, 110)


if __name__ == '__main__':
    main()
