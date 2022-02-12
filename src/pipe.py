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


def get_key_pose(file, frame):
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
    plt.legend(loc = "best")
    plt.xlabel('frames')
    plt.ylabel('errors')
    plt.show()


def write_predict_bvh(results, out):
    npa = np.asarray(results, dtype=np.float32)
    mpsk = MediaPipeSkeleton()
    channel, header = mpsk.poses2bvh(npa, output_file=out)
    print('write predicted bvh file,', out)
    ...


def save_mp_result():
    ...


def main():
    #######################################################
    video_file = 'ropejump.mp4'
    std_bvh_file = 'amassJumpRope0012json_rot_adjusted.bvh'
    key_pose_frame = 50
    output_file = 'ropejump_mp_output.bvh'
    #######################################################

    # start camera or import video
    cap = cv2.VideoCapture(video_file)
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

    # get one standard pose at frame
    std_pose = get_key_pose(std_bvh_file, key_pose_frame)

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

    all_array = []
    all_error_sum = []
    error_list_each_joint = [[] for y in range(len(joint_list))]

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
            for i, ele in enumerate(joint_list):
                error_list_each_joint[i].append(0)
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
            loc = joint_group[joint_name]
            arr.append(loc)

        npa = np.asarray([arr], dtype=np.float32)
        mpsk = MediaPipeSkeleton()
        channel, header = mpsk.poses2bvh(npa)
        channel = np.array(channel[0][3:])  # remove first 3 position, (51,)
        actual_channels = np.reshape(np.ravel(channel), (17, 3))

        ##################################
        # print('actual_channels at frame', frame_num)
        # print(np.cos(actual_channels), actual_channels.shape)
        ##################################

        # error for each joint XYZ
        for i, ele in enumerate(joint_list):
            error_sum_one_joint = calculate_sse(
                std_pose[i], actual_channels[i])
            error_list_each_joint[i].append(error_sum_one_joint)

        # error for 17 joints
        error_sum = calculate_sse(std_pose, actual_channels)
        all_error_sum.append(error_sum)
        print(frame_num, error_sum, np.cos(std_pose[13]), np.cos(actual_channels[13]))
        all_array.append(arr)
        # break


    print('#'*80)
    print('predict video:', video_file)
    print('standard bvh:', std_bvh_file)
    print('#'*80)
    # draw_line(key_pose_frame, all_error_sum, 1, frame_num)
    for i in range(17):
        draw_17_joint_line(key_pose_frame, error_list_each_joint, 40, 60, i)
    # write_predict_bvh(all_array, output_file)


if __name__ == '__main__':
    main()
