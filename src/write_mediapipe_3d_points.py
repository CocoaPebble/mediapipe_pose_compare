import cv2
import mediapipe as mp
import numpy as np


def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' +
                           str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()

def write_anlges_to_disk(filename, angles):
    with open(filename, 'w') as f:
        for frame_angle in angles:
            for ang in frame_angle:
                f.write(str(ang) + ' ')
            f.write('\n')

def write_mp_on_video(input_vid, output_vid):
    ...

def run_mp():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

    video_file = r'data\yoga4.mp4'
    cap = cv2.VideoCapture(video_file)
    kpts_3d = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        frame_keypoints = []
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if i not in pose_keypoints:
                    continue
                kpts = [landmark.x, landmark.y, landmark.z]
                frame_keypoints.append(kpts)
        else:
            frame_keypoints = [-1, -1, -1] * len(pose_keypoints)
        kpts_3d.append(frame_keypoints)
    return kpts_3d

kpts_3d = run_mp()
kpts_3d = np.array(kpts_3d)
write_keypoints_to_disk('3d_points_output/kpts_3d.dat', kpts_3d)


# test_nparr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# write_keypoints_to_disk('3d_points_output/test.dat', test_nparr)
