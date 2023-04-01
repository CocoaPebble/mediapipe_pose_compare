import os
import cv2
import math
import mediapipe as mp
import numpy as np
from scipy.spatial import procrustes

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles


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


def calc_left_knee(landmark):
    left_knee = landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    left_hip = landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_ankle = landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

    angle = angle_2p_3d(left_hip, left_knee, left_ankle)
    print("Left Knee 3d Angle: ", angle)
    angle = round(angle, 2)
    return angle


def calc_left_knee_2d(landmark):
    left_knee = landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    left_hip = landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_ankle = landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

    # get vectors from knee to hip and knee to ankle
    vec1 = left_knee - left_hip
    vec2 = left_knee - left_ankle

    # get only x and y components
    vec1 = vec1[0:2]
    vec2 = vec2[0:2]

    # calculate 2d angle
    angle = np.arccos(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    )
    angle = np.degrees(angle)
    angle = round(angle, 2)
    print("Left Knee 2d Angle: ", angle)
    return angle


def calc_left_hip(landmark):
    left_hip = landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_shoulder = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_knee = landmark[mp_pose.PoseLandmark.LEFT_KNEE]

    angle = angle_2p_3d(left_shoulder, left_hip, left_knee)
    angle = round(angle, 2)
    print("Left Hip 3d Angle: ", angle)
    return angle


def calc_left_hip_flexion(landmark):
    left_hip = landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_shoulder = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_knee = landmark[mp_pose.PoseLandmark.LEFT_KNEE]

    angle = angle_2p_3d(left_shoulder, left_hip, left_knee)
    angle = round(angle, 2)
    print("Left Hip 3d Angle: ", angle)
    return angle


def calc_left_hip_2d(landmark):
    left_hip = landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_shoulder = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_knee = landmark[mp_pose.PoseLandmark.LEFT_KNEE]

    # get vectors from hip to shoulder and hip to knee
    vec1 = left_hip - left_shoulder
    vec2 = left_hip - left_knee

    # get only x and y components
    vec1 = vec1[0:2]
    vec2 = vec2[0:2]

    # calculate 2d angle
    angle = np.arccos(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    )
    angle = np.degrees(angle)
    angle = round(angle, 2)
    print("Left Hip 2d Angle: ", angle)
    return angle


def calc_right_knee(landmark):
    right_knee = landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    right_hip = landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    right_ankle = landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    angle = angle_2p_3d(right_hip, right_knee, right_ankle)
    angle = round(angle, 2)
    print("Right Knee 3d Angle: ", angle)
    return angle


def calc_right_hip(landmark):
    right_hip = landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    right_shoulder = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_knee = landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

    angle = angle_2p_3d(right_shoulder, right_hip, right_knee)
    angle = round(angle, 2)
    print("Right Hip 3d Angle: ", angle)
    return angle


def calc_left_shoulder(landmark):
    left_shoulder = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_hip = landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_elbow = landmark[mp_pose.PoseLandmark.LEFT_ELBOW]

    angle = angle_2p_3d(left_hip, left_shoulder, left_elbow)
    angle = round(angle, 2)
    print("Left Shoulder 3d Angle: ", angle)
    return angle


def calc_right_shoulder(landmark):
    right_shoulder = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_hip = landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    right_elbow = landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

    angle = angle_2p_3d(right_hip, right_shoulder, right_elbow)
    angle = round(angle, 2)
    print("Right Shoulder 3d Angle: ", angle)
    return angle


def calc_left_elbow(landmark):
    left_elbow = landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_shoulder = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_wrist = landmark[mp_pose.PoseLandmark.LEFT_WRIST]

    angle = angle_2p_3d(left_shoulder, left_elbow, left_wrist)
    angle = round(angle, 2)
    print("Left Elbow 3d Angle: ", angle)
    return angle


def calc_right_elbow(landmark):
    right_elbow = landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_shoulder = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_wrist = landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    angle = angle_2p_3d(right_shoulder, right_elbow, right_wrist)
    angle = round(angle, 2)
    print("Right Elbow 3d Angle: ", angle)
    return angle


pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

np.set_printoptions(suppress=True)
myvideoname = "lying_leg_raise"
myvideopath = r"video/scoli-vid/" + myvideoname + ".mp4"
ref_frame = 400
cap = cv2.VideoCapture(myvideopath)
print("Start processing video " + myvideopath)
if not cap.isOpened():
    print("Error opening video stream or file.")
    raise TypeError
frame = 0
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(
    "cap_height: ",
    cap_height,
    "cap_width: ",
    cap_width,
    "fps: ",
    cap.get(cv2.CAP_PROP_FPS),
    "\n",
)

previous_joint_positions = np.zeros((33, 3))
next_joint_positions = np.zeros((33, 3))


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


mean_distance_arr = []
sum_distance_arr = []
disparity_arr = []
mean_similarity_arr = []
sum_similarity_arr = []
angle_arr = []

previous_left_knee_angle = 0
previous_right_knee_angle = 0
previous_left_hip_angle = 0
previous_right_hip_angle = 0
previous_left_shoulder_angle = 0
previous_right_shoulder_angle = 0
previous_left_elbow_angle = 0
previous_right_elbow_angle = 0
pre_angles = []

next_left_knee_angle = 0
next_right_knee_angle = 0
next_left_hip_angle = 0
next_right_hip_angle = 0
next_left_shoulder_angle = 0
next_right_shoulder_angle = 0
next_left_elbow_angle = 0
next_right_elbow_angle = 0

angle_diff = []

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    frame += 1
    if frame < ref_frame:
        continue
    print("Frame: ", frame)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if not results.pose_landmarks:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    cv2.putText(
        image,
        "Frame: " + str(frame),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("MediaPipe Pose", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

    ###############################
    landmarks = results.pose_landmarks.landmark
    landmarks_3d = results.pose_world_landmarks.landmark

    pose3d_landmark = np.array(
        [
            [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility,
            ]
            for landmark in landmarks_3d
        ]
    ).flatten()
    pose3d_landmark.resize((33, 4))

    left_knee_angle = calc_left_knee(pose3d_landmark)
    right_knee_angle = calc_right_knee(pose3d_landmark)
    left_hip_angle = calc_left_hip(pose3d_landmark)
    right_hip_angle = calc_right_hip(pose3d_landmark)
    left_shoulder_angle = calc_left_shoulder(pose3d_landmark)
    right_shoulder_angle = calc_right_shoulder(pose3d_landmark)
    left_elbow_angle = calc_left_elbow(pose3d_landmark)
    right_elbow_angle = calc_right_elbow(pose3d_landmark)

    if frame == ref_frame:
        for i, landmark in enumerate(landmarks_3d):
            previous_joint_positions[i] = [landmark.x, landmark.y, landmark.z]

        previous_left_knee_angle = left_knee_angle
        previous_right_knee_angle = right_knee_angle
        previous_left_hip_angle = left_hip_angle
        previous_right_hip_angle = right_hip_angle
        previous_left_shoulder_angle = left_shoulder_angle
        previous_right_shoulder_angle = right_shoulder_angle
        previous_left_elbow_angle = left_elbow_angle
        previous_right_elbow_angle = right_elbow_angle

        pre_angles = [
            left_knee_angle,
            right_knee_angle,
            left_hip_angle,
            right_hip_angle,
            left_shoulder_angle,
            right_shoulder_angle,
            left_elbow_angle,
            right_elbow_angle,
        ]

        # if not have the file to write, create it
        if not os.path.exists("log"):
            os.makedirs("log")

        # log the joint positions and angles in file for reference
        # with open("log/" + myvideoname + "_ref.txt", "w") as f:
        #     f.write("Frame: " + str(frame) + "\n")
        #     f.write("Joint positions: \n")
        #     f.write(str(previous_joint_positions))
        #     f.write("angles: \n")
        #     f.write(str(pre_angles))

    if frame > ref_frame:
        for i, landmark in enumerate(landmarks_3d):
            next_joint_positions[i] = [landmark.x, landmark.y, landmark.z]

        # Euclidean distance
        distances = [
            euclidean_distance(previous_joint_positions[i], next_joint_positions[i])
            for i in range(previous_joint_positions.shape[0])
        ]

        mean_distance = np.mean(distances)
        sum_distance = np.sum(distances)
        print("Mean distance:", mean_distance)
        print("Sum distance:", sum_distance)

        # Procrustes distance
        mtx1, mtx2, disparity = procrustes(
            previous_joint_positions, next_joint_positions
        )
        print("Disparity:", disparity)

        # Cosine similarity
        vector_diff = previous_joint_positions - next_joint_positions
        previous_joint_positions_norm = previous_joint_positions / np.linalg.norm(
            previous_joint_positions, axis=1, keepdims=True
        )
        next_joint_positions_norm = next_joint_positions / np.linalg.norm(
            next_joint_positions, axis=1, keepdims=True
        )
        similarities = [
            np.dot(previous_joint_positions_norm[i], next_joint_positions_norm[i])
            for i in range(previous_joint_positions_norm.shape[0])
        ]
        mean_similarity = np.mean(similarities)
        sum_similarity = np.sum(similarities)

        print("Mean similarity:", mean_similarity)
        print("Sum similarity:", sum_similarity)

        ###########################

        # Save to array
        mean_distance_arr.append(mean_distance)
        sum_distance_arr.append(sum_distance)
        disparity_arr.append(disparity)
        mean_similarity_arr.append(mean_similarity)
        sum_similarity_arr.append(sum_similarity)

        next_left_knee_angle = left_knee_angle
        next_right_knee_angle = right_knee_angle
        next_left_hip_angle = left_hip_angle
        next_right_hip_angle = right_hip_angle
        next_left_shoulder_angle = left_shoulder_angle
        next_right_shoulder_angle = right_shoulder_angle
        next_left_elbow_angle = left_elbow_angle
        next_right_elbow_angle = right_elbow_angle

        left_knee_angle_diff = next_left_knee_angle - previous_left_knee_angle
        right_knee_angle_diff = next_right_knee_angle - previous_right_knee_angle
        left_hip_angle_diff = next_left_hip_angle - previous_left_hip_angle
        right_hip_angle_diff = next_right_hip_angle - previous_right_hip_angle
        left_shoulder_angle_diff = (
            next_left_shoulder_angle - previous_left_shoulder_angle
        )
        right_shoulder_angle_diff = (
            next_right_shoulder_angle - previous_right_shoulder_angle
        )
        left_elbow_angle_diff = next_left_elbow_angle - previous_left_elbow_angle
        right_elbow_angle_diff = next_right_elbow_angle - previous_right_elbow_angle
        # print("angle diff:", left_knee_angle_diff, right_knee_angle_diff, left_hip_angle_diff, right_hip_angle_diff, left_shoulder_angle_diff, right_shoulder_angle_diff, left_elbow_angle_diff, right_elbow_angle_diff)

        # if either angle diff is greater than 5, then print the angle name and diff
        if abs(left_knee_angle_diff) > 5:
            print("left knee angle diff:", left_knee_angle_diff)
        if abs(right_knee_angle_diff) > 5:
            print("right knee angle diff:", right_knee_angle_diff)
        if abs(left_hip_angle_diff) > 5:
            print("left hip angle diff:", left_hip_angle_diff)
        if abs(right_hip_angle_diff) > 5:
            print("right hip angle diff:", right_hip_angle_diff)
        if abs(left_shoulder_angle_diff) > 5:
            print("left shoulder angle diff:", left_shoulder_angle_diff)
        if abs(right_shoulder_angle_diff) > 5:
            print("right shoulder angle diff:", right_shoulder_angle_diff)
        if abs(left_elbow_angle_diff) > 5:
            print("left elbow angle diff:", left_elbow_angle_diff)
        if abs(right_elbow_angle_diff) > 5:
            print("right elbow angle diff:", right_elbow_angle_diff)

        # calculate the diff sum with absolute value
        angle_diff_sum_abs = (
            abs(left_knee_angle_diff)
            + abs(right_knee_angle_diff)
            + abs(left_hip_angle_diff)
            + abs(right_hip_angle_diff)
            + abs(left_shoulder_angle_diff)
            + abs(right_shoulder_angle_diff)
            + abs(left_elbow_angle_diff)
            + abs(right_elbow_angle_diff)
        )
        print("angle diff sum abs:", angle_diff_sum_abs)

        # if the angle diff sum is greater than 40, then print pose is not correct
        if angle_diff_sum_abs > 40:
            print("pose is not correct")

        angle_diff.append(angle_diff_sum_abs)

    if frame == ref_frame + 300:
        # plot the line chart for array
        import matplotlib.pyplot as plt

        # create three subplots
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)
        ax1.plot(mean_distance_arr)
        ax1.legend(["mean_distance"])
        ax2.plot(sum_distance_arr)
        ax2.legend(["sum_distance"])
        ax3.plot(disparity_arr)
        ax3.legend(["disparity"])
        ax4.plot(mean_similarity_arr)
        ax4.legend(["mean_similarity"])
        ax5.plot(sum_similarity_arr)
        ax5.legend(["sum_similarity"])
        ax6.plot(angle_diff)
        ax6.legend(["angle_diff"])
        # plt.legend(["mean_distance", "sum_distance", "disparity", "mean_similarity", "sum_similarity"])
        plt.show()

    print("#" * 50)
