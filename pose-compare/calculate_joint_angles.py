import cv2
import math
import mediapipe as mp
import numpy as np
from scipy.spatial import procrustes
from time import sleep

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


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


def translation_normalization(landmark3d):
    left_hip = landmark3d[mp_pose.PoseLandmark.LEFT_HIP][:3]
    right_hip = landmark3d[mp_pose.PoseLandmark.RIGHT_HIP][:3]
    print("Left Hip: ", left_hip)
    print("Right Hip: ", right_hip)
    pelvis = (left_hip + right_hip) / 2
    
    print("Pelvis trans: ", pelvis)
    print("\n")

    for i in range(len(landmark3d)):
        joint_name = mp_pose.PoseLandmark(i)
        landmark3d[i][:3] = landmark3d[i][:3] - pelvis
        print(joint_name, ": ", landmark3d[i][:3])
    
    return landmark3d


def scale_normalization(landmark3d):
    left_hip = landmark3d[mp_pose.PoseLandmark.LEFT_HIP][:3]
    right_hip = landmark3d[mp_pose.PoseLandmark.RIGHT_HIP][:3]
    # calculte distance between left and right hip
    pelvis = np.linalg.norm(left_hip - right_hip)
    
    print("Pelvis scale: ", pelvis)
    print("\n")

    # scale all joints by pelvis distance
    for i in range(len(landmark3d)):
        landmark3d[i][:3] = landmark3d[i][:3] / pelvis
        print(landmark3d[i][:3])
        
    return landmark3d


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


# config for video
myvideoname = "bird_dog"
pose_number = 2
ref_frame = 570
exer_id = 28
myvideopath = r"video/scoli-vid/" + myvideoname + ".mp4"
# jsonfilepath = r"video/scoli-vid/" + myvideoname + "_" + str(pose_number) + ".json"
next_frame = ref_frame + 10

cap = cv2.VideoCapture(myvideopath)
print("Start processing video " + myvideopath)

if not cap.isOpened():
    print("Error opening video stream or file.")
    raise TypeError

frame = 0
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("cap_height: ", cap_height, "cap_width: ", cap_width, "fps: ", cap.get(cv2.CAP_PROP_FPS), "\n")

previous_joint_positions = np.zeros((33, 3))
next_joint_positions = np.zeros((33, 3))

previous_left_knee_angle = 0
previous_right_knee_angle = 0
previous_left_hip_angle = 0
previous_right_hip_angle = 0
previous_left_shoulder_angle = 0
previous_right_shoulder_angle = 0
previous_left_elbow_angle = 0
previous_right_elbow_angle = 0

next_left_knee_angle = 0
next_right_knee_angle = 0
next_left_hip_angle = 0
next_right_hip_angle = 0
next_left_shoulder_angle = 0
next_right_shoulder_angle = 0
next_left_elbow_angle = 0
next_right_elbow_angle = 0

ref_image = None

# using an array to store the angle diff sum after ref frame
angle_diff = []

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    frame += 1
    if frame < ref_frame - 3:
        continue
    print("Frame: ", frame)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if not results.pose_landmarks:
        continue

    ###############################
    pose2d = results.pose_landmarks
    pose3d = results.pose_world_landmarks

    if frame == ref_frame:
        # pose landmark to json
        json_data = {}
        json_data["exercise_id"] = exer_id
        json_data["exercise_name"] = myvideoname
        json_data["exercise_description"] = "Holding ladder and squat, finish position"
        json_data["exercise_image"] = myvideopath
        json_data["exercise_video"] = myvideopath
        json_data["video_height"] = cap_height
        json_data["video_width"] = cap_width
        json_data["keypose_number"] = pose_number
        json_data["ref_frame"] = frame

        json_data["2d_landmark"] = {}
        for i, landmark in enumerate(pose2d.landmark):
            json_data["2d_landmark"][mp_pose.PoseLandmark(i).name] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }
        # print("2d_landmark: ", json_data["2d_landmark"])

        json_data["3d_landmark"] = {}
        for i, landmark in enumerate(pose3d.landmark):
            json_data["3d_landmark"][mp_pose.PoseLandmark(i).name] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }
        # print("3d_landmark: ", json_data["3d_landmark"])

    pose2d_landmark = np.array(
        [
            [
                # get the pixel coordinate of the landmark
                landmark.x * cap_width,
                landmark.y * cap_height,
                landmark.z * cap_width,
                landmark.visibility,
            ]
            for landmark in pose2d.landmark
        ]
    ).flatten()
    pose2d_landmark.resize((33, 4))

    pose3d_landmark = np.array(
        [
            [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility,
            ]
            for landmark in pose3d.landmark
        ]
    ).flatten()
    pose3d_landmark.resize((33, 4))

    # translation_normalization(pose3d_landmark)

    # left_knee_angle = calc_left_knee_2d(pose2d_landmark)
    # left_hip_angle = calc_left_hip_2d(pose2d_landmark)

    left_knee_angle = calc_left_knee(pose3d_landmark)
    right_knee_angle = calc_right_knee(pose3d_landmark)
    left_hip_angle = calc_left_hip(pose3d_landmark)
    right_hip_angle = calc_right_hip(pose3d_landmark)
    left_shoulder_angle = calc_left_shoulder(pose3d_landmark)
    right_shoulder_angle = calc_right_shoulder(pose3d_landmark)
    left_elbow_angle = calc_left_elbow(pose3d_landmark)
    right_elbow_angle = calc_right_elbow(pose3d_landmark)

    if frame == ref_frame:
        
        print("", results.pose_world_landmarks)
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
        previous_left_knee_angle = left_knee_angle
        previous_right_knee_angle = right_knee_angle
        previous_left_hip_angle = left_hip_angle
        previous_right_hip_angle = right_hip_angle
        previous_left_shoulder_angle = left_shoulder_angle
        previous_right_shoulder_angle = right_shoulder_angle
        previous_left_elbow_angle = left_elbow_angle
        previous_right_elbow_angle = right_elbow_angle
        
        # update the previous joint positions
        for i, landmark in enumerate(pose3d.landmark):
            previous_joint_positions[i] = [landmark.x, landmark.y, landmark.z]
            # change the unit from meter to centimeter
            previous_joint_positions[i] = [x * 100 for x in previous_joint_positions[i]]
            
        print("previous_joint_positions: ", previous_joint_positions)
        
    if frame > ref_frame:
        # if frame == next_frame:
        next_left_knee_angle = left_knee_angle
        next_right_knee_angle = right_knee_angle
        next_left_hip_angle = left_hip_angle
        next_right_hip_angle = right_hip_angle
        next_left_shoulder_angle = left_shoulder_angle
        next_right_shoulder_angle = right_shoulder_angle
        next_left_elbow_angle = left_elbow_angle
        next_right_elbow_angle = right_elbow_angle

        # calculate the angle difference between the current frame and the next frame
        left_knee_angle_diff = next_left_knee_angle - previous_left_knee_angle
        right_knee_angle_diff = next_right_knee_angle - previous_right_knee_angle
        left_hip_angle_diff = next_left_hip_angle - previous_left_hip_angle
        right_hip_angle_diff = next_right_hip_angle - previous_right_hip_angle
        left_shoulder_angle_diff = next_left_shoulder_angle - previous_left_shoulder_angle
        right_shoulder_angle_diff = next_right_shoulder_angle - previous_right_shoulder_angle
        left_elbow_angle_diff = next_left_elbow_angle - previous_left_elbow_angle
        right_elbow_angle_diff = next_right_elbow_angle - previous_right_elbow_angle
        print("angle diff:", left_knee_angle_diff, right_knee_angle_diff, left_hip_angle_diff, right_hip_angle_diff, left_shoulder_angle_diff, right_shoulder_angle_diff, left_elbow_angle_diff, right_elbow_angle_diff)

        # update the next joint positions
        for i, landmark in enumerate(pose3d.landmark):
            next_joint_positions[i] = [landmark.x, landmark.y, landmark.z]
            # change the unit from meter to centimeter
            next_joint_positions[i] = next_joint_positions[i] * 100
        print("next_joint_positions: ", next_joint_positions)

        # calculate the distance between the current frame and the next frame
        mtx1, mtx2, disparity = procrustes(previous_joint_positions, next_joint_positions)
        np.set_printoptions(suppress=True)
        print("Matrix 1:\n", mtx1)
        print("Matrix 2:\n", mtx2)
        print("Disparity:", disparity)
        
        # calculate the diff sum with absolute value
        angle_diff_sum_abs = abs(left_knee_angle_diff) + abs(right_knee_angle_diff) + abs(left_hip_angle_diff) + abs(right_hip_angle_diff) + \
            abs(left_shoulder_angle_diff) + abs(right_shoulder_angle_diff) + \
            abs(left_elbow_angle_diff) + abs(right_elbow_angle_diff)
        print("angle diff sum abs:", angle_diff_sum_abs)
        
        angle_diff.append(angle_diff_sum_abs)

        # 

    # if frame == ref_frame:
    #     json_data["key_angle"] = {}
    #     json_data["key_angle"]["left_hip"] = left_hip_angle
    #     json_data["key_angle"]["right_hip"] = right_hip_angle
    #     json_data["key_angle"]["left_knee"] = left_knee_angle
    #     json_data["key_angle"]["right_knee"] = right_knee_angle
    #     json_data["key_angle"]["left_shoulder"] = left_shoulder_angle
    #     json_data["key_angle"]["right_shoulder"] = right_shoulder_angle
    #     json_data["key_angle"]["left_elbow"] = left_elbow_angle
    #     json_data["key_angle"]["right_elbow"] = right_elbow_angle

        # write json_data to file
        # with open(jsonfilepath, "w") as outfile:
        #     json.dump(json_data, outfile, indent=4)

        # sleep(5)
        # break

    ###############################
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    cv2.putText(image, "Frame: " + str(frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(image, "left knee: " + str(left_knee_angle), (10, 60),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(image, "left hip: " + str(left_hip_angle), (10, 90),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("MediaPipe Pose", image)
    
    if frame == ref_frame:
        ref_image = image
    
    if frame > ref_frame:
        # opencv show ref image
        cv2.imshow("ref image", ref_image)
    
    print("---------------------------")
    
    # when press 'y', display the chart of the angle difference
    if cv2.waitKey(1) & 0xFF == ord('y'):
        plt.plot(angle_diff)
        plt.show()

    if cv2.waitKey(5) & 0xFF == 27:
        break

pose.close()
cap.release()
