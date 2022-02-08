from . import math3d
from . import bvh_helper

import numpy as np


landmark_names = ['nose', 'left_eye_inner', 'left_eye',
                  'left_eye_outer', 'right_eye_inner', 'right_eye',
                  'right_eye_outer', 'left_ear', 'right_ear',
                  'mouth_left', 'mouth_right', 'left_shoulder',
                  'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_pinky_1',
                  'right_pinky_1', 'left_index_1', 'right_index_1',
                  'left_thumb_2', 'right_thumb_2', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                  'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']


class MediaPipeSkeleton(object):

    def __init__(self):

        # Define root of MediaPipe 3D pose data
        self.root = 'mid_hip'

        # Create the dictionary of keypoint and index according to MediaPipe
        self.keypoint2index = {
            'mid_hip': 0,
            'right_hip': 1,
            'right_knee': 2,
            'right_ankle': 3,
            'right_foot_index': 4,
            'left_hip': 5,
            'left_knee': 6,
            'left_ankle': 7,
            'left_foot_index': 8,
            'spine': 9,
            'mid_shoulder': 10,
            'left_shoulder': 11,
            'left_elbow': 12,
            'left_wrist': 13,
            'right_shoulder': 14,
            'right_elbow': 15,
            'right_wrist': 16,

            'right_foot_EndSite': -1,
            'left_foot_EndSite': -1,
            'left_wrist_EndSite': -1,
            'right_wrist_EndSite': -1,
            'mid_shoulder_EndSite': -1
        }

        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        # Create the parent-children dictionary
        self.children = {
            'mid_hip': ['left_hip', 'spine', 'right_hip'],
            'left_hip': ['left_knee'],
            'left_knee': ['left_ankle'],
            'left_ankle': ['left_foot_index'],
            'left_foot_index': ['left_foot_EndSite'],
            'left_foot_EndSite': [],
            'spine': ['mid_shoulder'],
            'mid_shoulder': ['left_shoulder', 'right_shoulder', 'mid_shoulder_EndSite'],
            'mid_shoulder_EndSite': [],
            'left_shoulder': ['left_elbow'],
            'left_elbow': ['left_wrist'],
            'left_wrist': ['left_wrist_EndSite'],
            'left_wrist_EndSite': [],
            'right_shoulder': ['right_elbow'],
            'right_elbow': ['right_wrist'],
            'right_wrist': ['right_wrist_EndSite'],
            'right_wrist_EndSite': [],
            'right_hip': ['right_knee'],
            'right_knee': ['right_ankle'],
            'right_ankle': ['right_foot_index'],
            'right_foot_index': ['right_foot_EndSite'],
            'right_foot_EndSite': []
        }

        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent

        self.left_joints = [
            joint for joint in self.keypoint2index
            if 'left_' in joint
        ]
        self.right_joints = [
            joint for joint in self.keypoint2index
            if 'right_' in joint
        ]

        # Create T-pose and define the direction from parent to child joint
        # What if the direction is not aligned with any axis????? I don't know, may be use sin, cos to represent it???????????????????
        self.initial_directions = {
            'mid_hip': [0, 0, 0],
            'right_hip': [-1, 0, 0],
            'right_knee': [0, 0, -1],
            'right_ankle': [0, 0, -1],
            'right_foot_index': [0, -1, 0],
            'right_foot_EndSite': [0, -1, 0],
            'left_hip': [1, 0, 0],
            'left_knee': [0, 0, -1],
            'left_ankle': [0, 0, -1],
            'left_foot_index': [0, -1, 0],
            'left_foot_EndSite': [0, -1, 0],
            'spine': [0, 0, 1],
            'mid_shoulder': [0, 0, 1],
            'mid_shoulder_EndSite': [0, 0, 1],
            'left_shoulder': [1, 0, 0],
            'left_elbow': [1, 0, 0],
            'left_wrist': [1, 0, 0],
            'left_wrist_EndSite': [1, 0, 0],
            'right_shoulder': [-1, 0, 0],
            'right_elbow': [-1, 0, 0],
            'right_wrist': [-1, 0, 0],
            'right_wrist_EndSite': [-1, 0, 0]
        }

    def get_initial_offset(self, poses_3d):
        bone_lens = {self.root: [0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            for child in self.children[parent]:
                if 'EndSite' in child:
                    bone_lens[child] = 0.4 * bone_lens[parent]
                    continue
                stack.append(child)

                c_idx = self.keypoint2index[child]
                bone_lens[child] = np.linalg.norm(
                    poses_3d[:, p_idx] - poses_3d[:, c_idx],
                    axis=1
                )

        bone_len = {}
        for joint in self.keypoint2index:
            if 'left_' in joint or 'right_' in joint:
                base_name = joint.replace('left_', '').replace('right_', '')
                left_len = np.mean(bone_lens['left_' + base_name])
                right_len = np.mean(bone_lens['right_' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / \
                max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset

    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)
        print(initial_offset)
        nodes = {}
        for joint in self.keypoint2index:
            print(joint)
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )
        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        header = bvh_helper.BvhHeader(root=nodes[self.root], nodes=nodes)
        return header

    # I think the most important part is here, to know the details of them please check the answer to issues in the GitHub https://github.com/KevinLTT/video2bvh/issues
    # I only adapt the code to our MediaPipe

    def pose2euler(self, pose, header):
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]

            if node.is_root:
                channel.extend(pose[joint_idx])

            index = self.keypoint2index
            order = None
            if joint == 'mid_hip':
                x_dir = pose[index['left_hip']] - pose[index['right_hip']]
                y_dir = None
                z_dir = pose[index['spine']] - pose[joint_idx]
                order = 'zyx'
            elif joint in ['right_hip', 'right_knee']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['mid_hip']] - pose[index['right_hip']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint in ['right_ankle']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = None
                y_dir = pose[joint_idx] - pose[child_idx]
                z_dir = pose[index['right_knee']] - pose[index['right_ankle']]
                order = 'yxz'
            elif joint in ['left_ankle']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = None
                y_dir = pose[joint_idx] - pose[child_idx]
                z_dir = pose[index['left_knee']] - pose[index['left_ankle']]
                order = 'yxz'
            elif joint in ['left_hip', 'left_knee']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['left_hip']] - pose[index['mid_hip']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint == 'spine':
                x_dir = pose[index['left_hip']] - pose[index['right_hip']]
                y_dir = None
                z_dir = pose[index['mid_shoulder']] - pose[joint_idx]
                order = 'zyx'
            elif joint == 'mid_shoulder':
                x_dir = pose[index['left_shoulder']] - \
                    pose[index['right_shoulder']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[index['spine']]
                order = 'zyx'
            elif joint == 'left_shoulder':
                x_dir = pose[index['left_elbow']] - pose[joint_idx]
                y_dir = pose[index['left_elbow']] - pose[index['left_wrist']]
                z_dir = None
                order = 'xzy'
            elif joint == 'left_elbow':
                x_dir = pose[index['left_wrist']] - pose[joint_idx]
                y_dir = pose[joint_idx] - pose[index['left_shoulder']]
                z_dir = None
                order = 'xzy'
            elif joint == 'right_shoulder':
                x_dir = pose[joint_idx] - pose[index['right_elbow']]
                y_dir = pose[index['right_elbow']] - pose[index['right_wrist']]
                z_dir = None
                order = 'xzy'
            elif joint == 'right_elbow':
                x_dir = pose[joint_idx] - pose[index['right_wrist']]
                y_dir = pose[joint_idx] - pose[index['right_shoulder']]
                z_dir = None
                order = 'xzy'
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )

            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.get_bvh_header(poses_3d)

        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))

        if output_file:
            bvh_helper.write_bvh(output_file, header, channels)

        return channels, header
