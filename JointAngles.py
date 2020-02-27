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

            # filtered keypoints

            if self.side == 'right':
                 parts = [[posture.joint_keypoints['RSHOULDER'], posture.joint_keypoints['RELBOW'],
                             posture.joint_keypoints['RWRIST'], posture.joint_keypoints['RHIP'],
                             posture.joint_keypoints['RKNEE'], posture.joint_keypoints['NECK'],
                             posture.joint_keypoints['MIDHIP']] for posture in frame_poses if posture.joint_keypoints['RSHOULDER'][2]!=0 and \
                             posture.joint_keypoints['RELBOW'][2]!=0 and posture.joint_keypoints['RWRIST'][2]!=0 and posture.joint_keypoints['RHIP'][2]!=0 and \
                             posture.joint_keypoints['RKNEE'][2]!=0 and posture.joint_keypoints['MIDHIP'][2]!=0 and posture.joint_keypoints['NECK'][2]!=0]

            else:
                parts = [[posture.joint_keypoints['LSHOULDER'], posture.joint_keypoints['LELBOW'],
                             posture.joint_keypoints['LWRIST'], posture.joint_keypoints['LHIP'],
                             posture.joint_keypoints['LKNEE'], posture.joint_keypoints['NECK'],
                             posture.joint_keypoints['MIDHIP']] for posture in frame_poses if posture.joint_keypoints['LSHOULDER'][2]!=0 and \
                             posture.joint_keypoints['LELBOW'][2]!=0 and posture.joint_keypoints['LWRIST'][2]!=0 and posture.joint_keypoints['LHIP'][2]!=0 and \
                             posture.joint_keypoints['LKNEE'][2]!=0 and posture.joint_keypoints['MIDHIP'][2]!=0 and posture.joint_keypoints['NECK'][2]!=0]
                
               

            forearm_vects = get_forearm_vectors(parts) 
            print('Size 1: ' + str(len(forearm_vects)))
            # self.forearm_vects.append(forearm_vect)
            upArm_vects = get_upper_arm_vectors(parts)
            print('Size 2: ' + str(len(upArm_vects)))
            # self.upArm_vects.append(upArm_vect)
            self.upArm_forearm_angles = get_upper_arm_forearm_angles(upArm_vects, forearm_vects)

            trunk_vects = get_trunk_vectors(parts)
            print('Size 1: ' + str(len(trunk_vects)))
            #self.trunk_vects.append(trunk_vect)
            self.upArm_trunk_angles = get_upper_arm_trunk_angles(upArm_vects, trunk_vects) 
            knee_vects = get_knee_vects(parts)
            print('Size 2: ' + str(len(knee_vects)))
            #self.knee_vects.append(knee_vect)
            self.trunk_knee_angles = get_trunk_knee_angles(trunk_vects, knee_vects)

        elif string == 'shoulder press':
            self.side = 'front'
            self.left_upArm_vects, self.right_upArm_vects = [], []
            self.left_forearm_vects, self.right_forearm_vects = [], []
            self.left_forearm_vects, self.right_forearm_vects = [], []
            self.left_upArm_forearm_angles, self.right_upArm_forearm_angles = [], []
            self.left_upArm_trunk_angles, self.right_upArm_trunk_angles = [], []
            self.trunk_vects = []
            
            parts = [[posture.joint_keypoints['LSHOULDER'], posture.joint_keypoints['RSHOULDER'],
                         posture.joint_keypoints['LELBOW'], posture.joint_keypoints['RELBOW'],
                         posture.joint_keypoints['LWRIST'], posture.joint_keypoints['RWRIST'],
                         posture.joint_keypoints['NECK'], posture.joint_keypoints['MIDHIP']] for posture in frame_poses if posture.joint_keypoints['RSHOULDER'][2]!=0 and \
                         posture.joint_keypoints['RELBOW'][2]!=0 and posture.joint_keypoints['RWRIST'][2]!=0 and posture.joint_keypoints['RHIP'][2]!=0 and \
                         posture.joint_keypoints['RKNEE'][2]!=0 and posture.joint_keypoints['LSHOULDER'][2]!=0 and posture.joint_keypoints['LELBOW'][2]!=0 and \
                         posture.joint_keypoints['LWRIST'][2]!=0 and posture.joint_keypoints['LHIP'][2]!=0 and \
                         posture.joint_keypoints['LKNEE'][2]!=0 and posture.joint_keypoints['MIDHIP'][2]!=0 and posture.joint_keypoints['NECK'][2]!=0]
                

            self.left_forearm_vects, self.right_forearm_vects = get_forearm_vectors(parts, view='front')
            self.left_upArm_vects, self.right_upArm_vects = get_upper_arm_vectors(parts, view='front')
            self.trunk_vects = get_trunk_vectors(parts)

            self.left_upArm_forearm_angles, self.right_upArm_forearm_angles = get_upper_arm_forearm_angles(self.right_upArm_vects,
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


def get_upper_arm_vectors(parts, view='side'):
    if view == 'side':
        # [0]- x; [1] - y, [2] - c
        # Shoulder - Elbow
        upArm_vects = [[part[0][0] - part[1][0], part[0][1] - part[1][1]] for part in parts]
        return upArm_vects
    elif view == 'front':
        left_upArm_vects = [[part[0][0] - part[2][0], part[0][1] - part[2][1]] for part in parts]
        right_upArm_vects = [[part[1][0] - part[3][0], part[1][1] - part[3][1]] for part in parts]
        return left_upArm_vects, right_upArm_vects


def get_forearm_vectors(parts, view='side'):
    if view == 'side':
        # Wrist - Elbow
        forearm_vects = [[part[2][0] - part[1][0], part[2][1] - part[1][1]] for part in parts]
        return forearm_vects
    elif view == 'front':
        left_forearm_vects = [[part[4][0] - part[2][0], part[4][1] - part[2][1]] for part in parts]
        right_forearm_vects = [[part[5][0] - part[3][0], part[5][1] - part[3][1]] for part in parts]
        return left_forearm_vects, right_forearm_vects


# Does not need view change
def get_trunk_vectors(parts):
    # Neck - MidHip
    trunk_vects = [[part[5][0] - part[6][0], part[5][1] - part[6][1]] for part in parts]
    return trunk_vects


# For now front view not needed (no exercise)
def get_knee_vects(parts):
    # Knee - Hip
    knee_vects = [[part[4][0] - part[3][0], part[4][1] - part[3][1]] for part in parts]
    return knee_vects


def get_upper_arm_trunk_angles(trunk_vects, upper_arm_vects1, upper_arm_vects2=None):
    upArm_trunk_angle1 = [calc_angle(upper_arm_vect, trunk_vect) for upper_arm_vect, trunk_vect in zip(upper_arm_vects1, trunk_vects)]
    if upper_arm_vects2 is None:
        return upArm_trunk_angle1
    else:
        upArm_trunk_angle2 =[calc_angle(upper_arm_vect, trunk_vect) for upper_arm_vect, trunk_vect in zip(upper_arm_vects2, trunk_vects)]
        return upArm_trunk_angle2, upArm_trunk_angle1


def get_upper_arm_forearm_angles(upper_arm_vects1, forearm_vects1, upper_arm_vects2=None, forearm_vects2=None):
    upArm_forearm_angle1 = [calc_angle(upper_arm_vect, forearm_vect) for upper_arm_vect, forearm_vect in zip(upper_arm_vects1, forearm_vects1)]
    if upper_arm_vects2 is None and forearm_vects2 is None:
        return upArm_forearm_angle1
    else:
        upArm_forearm_angle2 = [calc_angle(upper_arm_vect, forearm_vect) for upper_arm_vect, forearm_vect in zip(upper_arm_vects2, forearm_vects2)]
        return upArm_forearm_angle2, upArm_forearm_angle1


# One side for now
def get_trunk_knee_angles(trunk_vects, knee_vects):
    trunk_knee_angles = [calc_angle(trunk_vect, knee_vect) for trunk_vect, knee_vect in zip(trunk_vects, knee_vects)]
    return trunk_knee_angles