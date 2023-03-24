import cv2
import math
import mediapipe as mp
import numpy as np

# image = cv2.imread("test.jpg")
# standard_poses = np.load("standard_poses.npy", allow_pickle=True)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles


def main():
    supervideofile = r"D:\scoli-vid\How to Do the Bird Dog Exercise _ Abs Workout.mp4"
    cap = cv2.VideoCapture(supervideofile)
    print("Start processing video " + supervideofile)

    if not cap.isOpened():
        print("Error opening video stream or file.")
        raise TypeError

    frame = 0
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("cap_height: ", cap_height, "cap_width: ", cap_width)

    # mediapipe process video
    while cap.isOpened():
        frame += 1
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pose = mp_pose.Pose(
            static_image_mode=False,  # True for static images, False for videos
            model_complexity=1,  # 0, 1, or 2
            smooth_landmarks=True,  # True for static images, False for videos
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        image.flags.writeable = False
        results = pose.process(image)

        if not results.pose_landmarks:
            continue

        print("frame: ", frame)
        pose2d = results.pose_landmarks
        pose3d = results.pose_world_landmarks

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            pose2d,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        pose_landmark = np.array(
            [
                [
                    landmark.x * cap_width,
                    landmark.y * cap_height,
                    landmark.z * cap_width,
                    landmark.visibility,
                ]
                for landmark in pose2d.landmark
            ]
        ).flatten()
        pose_landmark.resize((33, 4))
        print(pose_landmark)

        left_knee_angle = calc_left_knee_angle(pose3d)
        right_knee_angle = calc_right_knee_angle(pose3d)
        left_elbow = calc_left_elbow_angle(pose3d)
        right_elbow = calc_right_elbow_angle(pose3d)

        leg_height = calc_body_standing_height(pose3d)
        full_body_height = calc_full_body_standing_height(pose3d)
        print("leg_height: ", leg_height)
        print("full_body_height: ", full_body_height)

        cv2.putText(
            image,
            "left_knee" + str(left_knee_angle),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "right_knee" + str(right_knee_angle),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "left_elbow" + str(left_elbow),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "right_elbow" + str(right_elbow),
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("MediaPipe Pose", image)
        print("---------------------------------")

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    pose.close()

def get_left_knee_angle(pose):
    left_knee = pose.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    left_hip = pose.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_ankle = pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    return calc_angle(left_hip, left_knee, left_ankle)


def analyze_standard_pose(pose):
    pose_landmark = np.array(
        [
            [landmark.x, landmark.y, landmark.z, landmark.visibility]
            for landmark in pose.landmark
        ]
    ).flatten()
    pose_landmark.resize((33, 4))
    # print(pose_landmark)
    left_knee_angle = calc_left_knee_angle(pose)
    right_knee_angle = calc_right_knee_angle(pose)
    left_elbow = calc_left_elbow_angle(pose)
    right_elbow = calc_right_elbow_angle(pose)
    print("left_knee_angle: ", left_knee_angle)
    print("right_knee_angle: ", right_knee_angle)
    print("left_elbow: ", left_elbow)
    print("right_elbow: ", right_elbow)

    # Calculate the average of each angle


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


def calc_eculidean_distance(pose1, pose2):
    landmark1 = np.array(
        ([landmark.x, landmark.y, landmark.z]) for landmark in pose1.landmark
    )
    landmark2 = np.array(
        ([landmark.x, landmark.y, landmark.z]) for landmark in pose2.landmark
    )

    distance = np.sqrt(np.sum((landmark1 - landmark2) ** 2, axis=1))
    eculidean_distance = np.mean(distance)
    return eculidean_distance


def calc_distance_between_two_joints(pose1, pose2, joint1, joint2):
    joint1_id = mp_pose.PoseLandmark(joint1).value
    joint2_id = mp_pose.PoseLandmark(joint2).value

    landmark1 = np.array(
        [
            pose1.landmark[joint1_id].x,
            pose1.landmark[joint1_id].y,
            pose1.landmark[joint1_id].z,
        ]
    )
    landmark2 = np.array(
        [
            pose2.landmark[joint2_id].x,
            pose2.landmark[joint2_id].y,
            pose2.landmark[joint2_id].z,
        ]
    )
    distance = np.sqrt(np.sum((landmark1 - landmark2) ** 2))
    return distance


def calc_body_standing_height(pose):
    # get the coords of the left hip and right hip
    left_hip = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z,
        ]
    )
    right_hip = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z,
        ]
    )
    # get the coords of the left ankle and right ankle
    left_ankle = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].z,
        ]
    )
    right_ankle = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z,
        ]
    )

    # get the distance between the left hip and left ankle
    left_hip_to_left_ankle = np.sqrt(np.sum((left_hip - left_ankle) ** 2))
    # get the distance between the right hip and right ankle
    right_hip_to_right_ankle = np.sqrt(np.sum((right_hip - right_ankle) ** 2))
    # get the average of the two distances
    body_standing_height = (left_hip_to_left_ankle + right_hip_to_right_ankle) / 2

    return body_standing_height


def calc_full_body_standing_height(pose):
    # get the coord of the left ankle and right ankle
    left_ankle = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].z,
        ]
    )
    right_ankle = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z,
        ]
    )

    # get the coord of the left shoulder and right shoulder
    left_shoulder = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
        ]
    )

    right_shoulder = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
        ]
    )

    # calculate the distance between the left ankle and left shoulder
    left_ankle_to_left_shoulder = np.sqrt(np.sum((left_ankle - left_shoulder) ** 2))
    # calculate the distance between the right ankle and right shoulder
    right_ankle_to_right_shoulder = np.sqrt(np.sum((right_ankle - right_shoulder) ** 2))

    # get the average of the two distances
    full_body_standing_height = (
        left_ankle_to_left_shoulder + right_ankle_to_right_shoulder
    ) / 2

    return full_body_standing_height


def calc_left_knee_angle(pose):
    # get the coords of the left knee, left hip, and left ankle

    left_knee = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].z,
        ]
    )
    left_hip = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z,
        ]
    )
    left_ankle = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].z,
        ]
    )

    # print('point left_knee: ', left_knee)
    # print('point left_hip: ', left_hip)
    # print('point left_ankle: ', left_ankle)

    # calculate the vectors between the left knee to left hip, and left knee to left ankle
    # hip_knee_vec = left_hip - left_knee
    # ankle_knee_vec = left_ankle - left_knee

    # calculate the angle between the vectors
    angle = angle_2p_3d(left_hip, left_knee, left_ankle)
    # print('func1, angle1: ', angle1)

    # dot_prod = np.dot(hip_knee_vec, ankle_knee_vec)
    # angle = np.degrees(np.arccos(dot_prod / (np.linalg.norm(hip_knee_vec) * np.linalg.norm(ankle_knee_vec))))
    # print('func2, angle: ', np.round(angle, 2))

    return angle


def calc_right_knee_angle(pose):
    # get the coords of the right knee, right hip, and right ankle
    right_knee = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].z,
        ]
    )
    right_hip = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z,
        ]
    )
    right_ankle = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z,
        ]
    )

    angle = angle_2p_3d(right_hip, right_knee, right_ankle)
    # print('right knee angle: ', angle)

    return angle


def calc_left_elbow_angle(pose):
    # get the coords of the left elbow, left shoulder, and left wrist
    left_elbow = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
        ]
    )
    left_shoulder = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
        ]
    )
    left_wrist = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
        ]
    )

    angle = angle_2p_3d(left_shoulder, left_elbow, left_wrist)
    # print('left elbow angle: ', angle, ' degrees')

    return angle


def calc_right_elbow_angle(pose):
    # get the coords of the right elbow, right shoulder, and right wrist
    right_elbow = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z,
        ]
    )
    right_shoulder = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
        ]
    )
    right_wrist = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            pose.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].z,
        ]
    )

    angle = angle_2p_3d(right_shoulder, right_elbow, right_wrist)
    # print('right elbow angle: ', angle)

    return angle


def calc_left_armpit_angle(pose):
    # get the coords of the left elbow, left shoulder, and left wrist
    left_shoulder = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
        ]
    )
    left_hip = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z,
        ]
    )
    left_wrist = np.array(
        [
            pose.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            pose.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            pose.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
        ]
    )

    angle = angle_2p_3d(left_shoulder, left_hip, left_wrist)
    # print('left armpit angle: ', angle)

    return angle


if __name__ == "__main__":
    main()
