import subprocess
import sys

try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "--user"])

import math
from Functions import detect_side


class JointAngles:
    def __init__(self, string, frame_pose):
        if string == 'bicep_curl' or string == 'front_raise' or string == 'triceps_pushdown':
            self.side, parts_filtered = detect_side(frame_pose)

            forearm_vects = get_forearm_vectors(parts_filtered)


            # self.forearm_vects.append(forearm_vect)
            upArm_vects = get_upper_arm_vectors(parts_filtered)
            # self.upArm_vects.append(upArm_vect)
            print(upArm_vects, forearm_vects)
            self.upArm_forearm_angles = get_upper_arm_forearm_angles(upArm_vects, forearm_vects)
            trunk_vects = get_trunk_vectors(parts_filtered)
            # self.trunk_vects.append(trunk_vect)
            self.upArm_trunk_angles = get_upper_arm_trunk_angles(upArm_vects, trunk_vects)
            knee_vects = get_knee_vects(parts_filtered)
            # self.knee_vects.append(knee_vect)
            self.trunk_knee_angles = get_trunk_knee_angles(trunk_vects, knee_vects, self.side)

        elif string == 'shoulder_press':
            self.side = 'front'
            joints = ['LSHOULDER', 'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST',
                      'NECK', 'MIDHIP']

            parts = [frame_pose.joint_keypoints[joint] for joint in joints]

            parts_filtered = [part for part in parts if all(joint_points[2] != 0 for joint_points in part)]

            self.left_forearm_vects, self.right_forearm_vects = get_forearm_vectors(parts_filtered, view='front')
            self.left_upArm_vects, self.right_upArm_vects = get_upper_arm_vectors(parts_filtered, view='front')
            self.trunk_vects = get_trunk_vectors(parts_filtered, view='front')

            self.left_upArm_forearm_angles, self.right_upArm_forearm_angles = get_upper_arm_forearm_angles(
                self.right_upArm_vects,
                self.right_forearm_vects,
                self.left_upArm_vects,
                self.left_forearm_vects)

            self.left_upArm_trunk_angles, self.right_upArm_trunk_angles = get_upper_arm_trunk_angles(self.trunk_vects,
                                                                                                     self.left_upArm_vects,
                                                                                                     self.right_upArm_vects)
        else:
            print("Either typed the exercise name wrong or typed some new exercise")


# Normalize vectors
def unit_vector(vector):
    # print('Vector :' + str(vector / np.linalg.norm(vector)))
    return vector / np.linalg.norm(vector)


# Calculate angle between vectors
def calc_angle(vect1, vect2):
    unit_vect1 = unit_vector(vect1)
    unit_vect2 = unit_vector(vect2)
    return math.degrees(np.arccos(np.clip(np.dot(unit_vect1, unit_vect2), -1.0, 1.0)))


def calc_tk_angle(vect1, vect2):
    unit_vect1 = unit_vector(vect1)
    unit_vect2 = unit_vector(vect2)

    angle = math.atan2(unit_vect1[1], unit_vect1[0]) - math.atan2(unit_vect2[1], unit_vect2[0])
    angle = angle * 360 / (2 * math.pi)
    if angle < 0:
        angle += 360

    return angle


def get_upper_arm_vectors(parts, view='side'):
    if not parts:
        # print('parts list empty')
        return None

    if view == 'side':
        # [0]- x; [1] - y, [2] - c
        # Shoulder - Elbow
        upArm_vects = [parts[0][0] - parts[1][0], parts[0][1] - parts[1][1]]
        return upArm_vects
    elif view == 'front':
        left_upArm_vects = [parts[0][0] - parts[2][0], parts[0][1] - parts[2][1]]
        right_upArm_vects = [parts[1][0] - parts[3][0], parts[1][1] - parts[3][1]]
        return left_upArm_vects, right_upArm_vects


def get_forearm_vectors(parts, view='side'):
    if not parts:
        # print('parts list empty')
        return None

    if view == 'side':
        # Wrist - Elbow
        forearm_vects = [parts[2][0] - parts[1][0], parts[2][1] - parts[1][1]]
        return forearm_vects
    elif view == 'front':
        left_forearm_vects = [parts[4][0] - parts[2][0], parts[4][1] - parts[2][1]]
        right_forearm_vects = [parts[5][0] - parts[3][0], parts[5][1] - parts[3][1]]
        return left_forearm_vects, right_forearm_vects


# Does not need view change
def get_trunk_vectors(parts, view='side'):
    if not parts:
        # print('parts list empty')
        return None

    if view == 'side':
        # Neck - MidHip
        trunk_vects = [parts[5][0] - parts[6][0], parts[5][1] - parts[6][1]]
        return trunk_vects
    elif view == 'front':
        trunk_vects = [parts[6][0] - parts[7][0], parts[6][1] - parts[7][1]]
        return trunk_vects


# For now front view not needed (no exercise)
def get_knee_vects(parts):
    if not parts:
        # print('parts list empty')
        return None

    # Knee - Hip
    knee_vects = [parts[4][0] - parts[3][0], parts[4][1] - parts[3][1]]
    return knee_vects


def get_upper_arm_trunk_angles(trunk_vects, upper_arm_vects1, upper_arm_vects2=None):
    upArm_trunk_angle1 = [calc_angle(upper_arm_vects1, trunk_vects)]
    if upper_arm_vects2 is None:
        return upArm_trunk_angle1
    else:
        upArm_trunk_angle2 = [calc_angle(upper_arm_vects2, trunk_vects)]
        return upArm_trunk_angle2, upArm_trunk_angle1


def get_upper_arm_forearm_angles(upper_arm_vects1, forearm_vects1, upper_arm_vects2=None, forearm_vects2=None):
    upArm_forearm_angle1 = [calc_angle(upper_arm_vects1, forearm_vects1)]

    if upper_arm_vects2 is None and forearm_vects2 is None:
        return upArm_forearm_angle1
    elif upper_arm_vects2 is not None and forearm_vects2 is not None:
        upArm_forearm_angle2 = [calc_angle(upper_arm_vects2, forearm_vects1)]
        return upArm_forearm_angle2, upArm_forearm_angle1
    else:
        return None


# One side for now
def get_trunk_knee_angles(trunk_vects, knee_vects, side):
    if side == 'right':
        trunk_knee_angles = [calc_tk_angle(knee_vects, trunk_vects)]
    elif side == 'left':
        trunk_knee_angles = [calc_tk_angle(trunk_vects, knee_vects)]
    else:
        return None

    return trunk_knee_angles
