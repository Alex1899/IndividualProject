import glob
import json
import os
import subprocess
import sys
from Frame import FramePose
import numpy as np

"""
try:

except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "--user"])
    import numpy as np
"""


def parse_frames(path_to_json):
    """
    Gets keypoints from each frame and passes them to FramePose class.
    :param path_to_jsons: path to the folder where json files extracted using OpenPose are located.
    :return: a list containing FramePose objects where all keypoints for each frame are saved
    """

    if not path_to_json:
        print("Error: could not obtain JSON files from the keypoints folder")
        print("Please, make sure the folder with the exercise video name is not empty, "
              "which is located in the keypoints folder, before running the system again. ")
        return None

    with open(path_to_json) as obj:
        file_json = json.load(obj)
        keypoints = np.array(file_json['people'][0]['pose_keypoints_2d'])
        pose = FramePose(keypoints.reshape((25, 3)))

    return pose


def distance(joint1, joint2):
    """
    Calculate distance between points of two joints
    :param joint1: first joint
    :param joint2: Second joint
    :return: distance between joints
    """
    dist = np.sqrt(np.square(joint1[0] - joint2[0]) + np.square((joint1[1] - joint2[1])))
    return dist


def normalise(pose, torso_mean):
    """
    Normalise keypoint values using mean torso length
    :param pose: a FramePose objects
    :param torso_mean: mean torso value
    """

    for key, value in pose.joint_keypoints.items():
        # print("old value x " + key + " " + str(value[0]) + " " + str(value[1]))
        # x value
        pose.joint_keypoints[key][0] = pose.joint_keypoints[key][0] / torso_mean
        # y value
        pose.joint_keypoints[key][1] = pose.joint_keypoints[key][1] / torso_mean

    return pose
