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
    supervideofile = "./video/squat1.mp4"
    cap = cv2.VideoCapture(supervideofile)
    print("Start processing video " + supervideofile)

    if not cap.isOpened():
        print("Error opening video stream or file.")
        raise TypeError

    # mediapipe process video
    while cap.isOpened():
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
        left_knee_angle = calc_left_knee_angle(pose2d)
        right_knee_angle = calc_right_knee_angle(pose2d)
        left_elbow = calc_left_elbow_angle(pose2d)
        right_elbow = calc_right_elbow_angle(pose2d)
        
        cv2.putText(image, 'left_knee' + str(left_knee_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'right_knee' + str(right_knee_angle), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'left_elbow'+str(left_elbow), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'right_elbow'+str(right_elbow), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("MediaPipe Pose", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    pose.close()


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

    # calculate the vectors between the joints
    v1 = left_knee - left_hip
    v2 = left_ankle - left_hip

    # calculate the angle between the vectors
    angle = angle_2p_3d(left_hip, left_knee, left_ankle)
    # print(angle)

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

    # calculate the vectors between the joints
    # v1 = right_knee - right_hip
    # v2 = right_ankle - right_hip

    # # calculate the angle between the vectors
    # angle = np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
    # angle = np.degrees(angle)
    
    angle = angle_2p_3d(right_hip, right_knee, right_ankle)
    print('right knee angle: ', angle)
    
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

    # # calculate the vectors between the joints
    # v1 = left_elbow - left_shoulder
    # v2 = left_wrist - left_shoulder

    # # calculate the angle between the vectors
    # angle = np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
    # angle = np.degrees(angle)
    
    angle = angle_2p_3d(left_shoulder, left_elbow, left_wrist)
    print('left elbow angle: ', angle, ' degrees')
    
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

    # # calculate the vectors between the joints
    # v1 = right_elbow - right_shoulder
    # v2 = right_wrist - right_shoulder

    # # calculate the angle between the vectors
    # angle = np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
    # angle = np.degrees(angle)
    
    angle = angle_2p_3d(right_shoulder, right_elbow, right_wrist)
    print('right elbow angle: ', angle)
    
    return angle


if __name__ == "__main__":
    main()
