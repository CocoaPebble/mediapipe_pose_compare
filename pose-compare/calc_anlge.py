import numpy as np
import math
import mediapipe as mp
mp_pose = mp.solutions.pose


# create a body angle class to calculate the angle of the mediapipe skeleton
class BodyAngle:
    
    def __init__(self, keypoints):
        self.keypoints = keypoints

    def angle_2p_3d(self, a, b, c):
        v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])

        v1mag = np.sqrt([v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]])
        v1norm = np.array([v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

        v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
        v2norm = np.array([v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])

        res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
        angle_rad = np.arccos(res)

        return round(math.degrees(angle_rad), 2)
    
    # get angle of the mediapipe pose body joints, left and right hip, left and right shoulder, left and right elbow, left and right knee
    def get_angle_of_body(self):
        angle_of_body = []
        left_hip_angle = self.angle_2p_3d(self.keypoints[mp_pose.PoseLandmark.LEFT_HIP.value], self.keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value], self.keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        right_hip_angle = self.angle_2p_3d(self.keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value], self.keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value], self.keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        left_shoulder_angle = self.angle_2p_3d(self.keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value], self.keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value], self.keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value])
        right_shoulder_angle = self.angle_2p_3d(self.keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], self.keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value], self.keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        left_knee_angle = self.angle_2p_3d(self.keypoints[mp_pose.PoseLandmark.LEFT_HIP.value], self.keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value], self.keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        right_knee_angle = self.angle_2p_3d(self.keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value], self.keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value], self.keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        left_elbow_angle = self.angle_2p_3d(self.keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value], self.keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value], self.keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value])
        right_elbow_angle = self.angle_2p_3d(self.keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], self.keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value], self.keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        angle_of_body.append(left_hip_angle)
        angle_of_body.append(right_hip_angle)
        angle_of_body.append(left_shoulder_angle)
        angle_of_body.append(right_shoulder_angle)
        angle_of_body.append(left_knee_angle)
        angle_of_body.append(right_knee_angle)
        angle_of_body.append(left_elbow_angle)
        angle_of_body.append(right_elbow_angle)
        
        return angle_of_body
    
    # calculate the angle diff sum abs of two body angles
    def angle_diff_sum_abs(self, angle1, angle2):
        angle_diff_sum_abs = 0
        for i in range(len(angle1)):
            angle_diff_sum_abs += abs(angle1[i] - angle2[i])
        
        return angle_diff_sum_abs
    
    # calculate the angle diff square root sum of two body angles
    def angle_diff_sqrt_sum(self, angle1, angle2):
        angle_diff_sqrt_sum = 0
        for i in range(len(angle1)):
            angle_diff_sqrt_sum += math.sqrt(abs(angle1[i] - angle2[i]))
        
        return angle_diff_sqrt_sum
    
    # calculate the angle diff mean of two body angles
    def angle_diff_mean(self, angle1, angle2):
        angle_diff_mean = 0
        for i in range(len(angle1)):
            angle_diff_mean += abs(angle1[i] - angle2[i])
        
        return angle_diff_mean / len(angle1)
    