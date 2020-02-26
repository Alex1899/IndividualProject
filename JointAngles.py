import numpy as np
import math
from scipy.signal import medfilt
from Functions import detect_side


class JointAngles:
    def __init__(self, string, frame_poses):
        if string == 'bicep curl' or string == 'front raise' or string == 'triceps pushdown':
            self.side = detect_side(frame_poses)
            self.forearm_vects, self.upArm_vects = [], []
            self.trunk_vects, self.knee_vects = [], []
            self.upArm_forearm_angles, self.upArm_trunk_angles = [], []
            self.trunk_knee_angles = []

            for posture in frame_poses:
                if self.side == 'right':
                    parts = [posture.joint_keypoints['RSHOULDER'], posture.joint_keypoints['RELBOW'],
                             posture.joint_keypoints['RWRIST'], posture.joint_keypoints['RHIP'],
                             posture.joint_keypoints['RKNEE'], posture.joint_keypoints['NECK'],
                             posture.joint_keypoints['MIDHIP']]
                else:
                    parts = [posture.joint_keypoints['LSHOULDER'], posture.joint_keypoints['LELBOW'],
                             posture.joint_keypoints['LWRIST'], posture.joint_keypoints['LHIP'],
                             posture.joint_keypoints['LKNEE'], posture.joint_keypoints['NECK'],
                             posture.joint_keypoints['MIDHIP']]
                
               
                # need to filter keypoints

                forearm_vect = get_forearm_vectors(parts)
                self.forearm_vects.append(forearm_vect)
                upArm_vect = get_upper_arm_vectors(parts)
                self.upArm_vects.append(upArm_vect)
                self.upArm_forearm_angles.append(get_upper_arm_forearm_angles(upArm_vect, forearm_vect))

                trunk_vect = get_trunk_vectors(parts)
                self.trunk_vects.append(trunk_vect)
                self.upArm_trunk_angles.append(get_upper_arm_trunk_angles(upArm_vect, trunk_vect))

                knee_vect = get_knee_vects(parts)
                self.knee_vects.append(knee_vect)
                self.trunk_knee_angles.append(get_trunk_knee_angles(trunk_vect, knee_vect))

        elif string == 'shoulder press':
            self.side = 'front'
            self.left_upArm_vects, self.right_upArm_vects = [], []
            self.left_forearm_vects, self.right_forearm_vects = [], []
            self.left_forearm_vects, self.right_forearm_vects = [], []
            self.left_upArm_forearm_angles, self.right_upArm_forearm_angles = [], []
            self.left_upArm_trunk_angles, self.right_upArm_trunk_angles = [], []
            self.trunk_vects = []

            for posture in frame_poses:
                parts = [posture.joint_keypoints['LSHOULDER'], posture.joint_keypoints['RSHOULDER'],
                         posture.joint_keypoints['LELBOW'], posture.joint_keypoints['RELBOW'],
                         posture.joint_keypoints['LWRIST'], posture.joint_keypoints['RWRIST'],
                         posture.joint_keypoints['NECK'], posture.joint_keypoints['MIDHIP']]

                left_fa, right_fa = get_forearm_vectors(parts, view='front')
                self.left_forearm_vects.append(left_fa)
                self.right_upArm_trunk_angles.append(right_fa)
                left_uav, right_uav = get_upper_arm_vectors(parts, view='front')
                self.left_upArm_vects.append(left_uav)
                self.right_forearm_vects.append(right_uav)
                self.trunk_vects = get_trunk_vectors(parts)
                # self.knee_vects = get_knee_vects(self.parts)
                left_forarm_angles, right_forearm_angles = get_upper_arm_forearm_angles(self.right_upArm_vects,
                                                                                        self.right_forearm_vects,
                                                                                        self.left_upArm_vects,
                                                                                        self.left_forearm_vects)

                self.left_upArm_forearm_angles.append(left_forarm_angles)
                self.right_upArm_forearm_angles.append(right_forearm_angles)
                left_trunk_angles, right_trunk_angles = get_upper_arm_trunk_angles(self.trunk_vects,
                                                                                   self.left_upArm_vects,
                                                                                   self.right_upArm_vects)

                self.left_upArm_trunk_angles.append(left_trunk_angles)
                self.right_upArm_trunk_angles.append(right_trunk_angles)
        else:
            print("Either typed the exercise name wrong or typed some new exercise")


# Normalize vectors
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


# Calculate angle between vectors
def calc_angle(vect1, vect2):
    unit_vect1 = unit_vector(vect1)
    unit_vect2 = unit_vector(vect2)
    return math.degrees(np.arccos(np.clip(np.dot(unit_vect1, unit_vect2), -1.0, 1.0)))


def get_upper_arm_vectors(parts, view='side'):
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
    if view == 'side':
        # Wrist - Elbow
        forearm_vects = [parts[2][0] - parts[1][0], parts[2][1] - parts[1][1]]
        return forearm_vects
    elif view == 'front':
        left_forearm_vects = [parts[4][0] - parts[2][0], parts[4][1] - parts[2][1]]
        right_forearm_vects = [parts[5][0] - parts[3][0], parts[5][1] - parts[3][1]]
        return left_forearm_vects, right_forearm_vects


# Does not need view change
def get_trunk_vectors(parts):
    # Neck - MidHip
    trunk_vects = [parts[5][0] - parts[6][0], parts[5][1] - parts[6][1]]
    return trunk_vects


# For now front view not needed (no exercise)
def get_knee_vects(parts):
    # Knee - Hip
    knee_vects = [parts[4][0] - parts[3][0], parts[4][1] - parts[3][1]]
    return knee_vects


def get_upper_arm_trunk_angles(trunk_vects, upper_arm_vects1, upper_arm_vects2=None):
    if upper_arm_vects2 is None:
        # Calculate angle
        right_upArm_trunk_angle = calc_angle(upper_arm_vects1, trunk_vects)
        # might use Kalman filter or something later instead
        return right_upArm_trunk_angle
    else:
        right_upArm_trunk_angle = calc_angle(upper_arm_vects1, trunk_vects)
        left_upArm_trunk_angle = calc_angle(upper_arm_vects2, trunk_vects)
        return left_upArm_trunk_angle, right_upArm_trunk_angle


def get_upper_arm_forearm_angles(upper_arm_vects1, forearm_vects1, upper_arm_vects2=None, forearm_vects2=None):
    if upper_arm_vects2 is None and forearm_vects2 is None:
        upArm_forearm_angle = calc_angle(upper_arm_vects1, forearm_vects1)
        # might use Kalman filter or something later instead
        return upArm_forearm_angle
    else:
        right_upArm_forearm_angle = calc_angle(upper_arm_vects1, forearm_vects1)
        left_upArm_forearm_angle = calc_angle(upper_arm_vects2, forearm_vects2)
        return left_upArm_forearm_angle, right_upArm_forearm_angle


# One side for now
def get_trunk_knee_angles(trunk_vects, knee_vects):
    trunk_knee_angle = calc_angle(trunk_vects, knee_vects)
    return trunk_knee_angle
