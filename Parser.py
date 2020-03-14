import glob
import json
import os
import subprocess
import sys
from Frame import FramePose

try:
    import numpy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "--user"])


def parse_frames(path_to_jsons):
    """
    Gets keypoints from each frame and passes them to FramePose class.
    :param path_to_jsons: path to the folder where json files extracted using OpenPose are located.
    :return: a list containing FramePose objects where all keypoints for each frame are saved
    """
   
    json_files = glob.glob(os.path.join(path_to_jsons, "*.json"))
    num_json_files = len(json_files)
    frame_poses = []
    for num in range(num_json_files):
        with open(json_files[num]) as obj:
            file_json = json.load(obj)
            if len(file_json['people']) > 0:
                keypoints = numpy.array(file_json['people'][0]['pose_keypoints_2d'])
                pose = FramePose(keypoints.reshape((25, 3)))
                frame_poses.append(pose)

    torso_values = numpy.array([])
    torso_values = numpy.append(torso_values,[distance(pose.joint_keypoints['NECK'], pose.joint_keypoints['MIDHIP']) for pose in frame_poses])
    mean_torso_value = numpy.mean(torso_values)
    
    frame_poses = [normalise(pose, mean_torso_value) for pose in frame_poses]
          
    return frame_poses


def distance(joint1, joint2):
    """
    Calculate distance between points of two joints
    :param joint1: first joint
    :param joint2: Second joint
    :return: distance between joints
    """
    dist = numpy.sqrt(numpy.square(joint1[0] - joint2[0]) + numpy.square((joint1[1] - joint2[1])))
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
    
     


"""
    for key, value in new_joint_keypoints.items():
        print("new value x " + key + " " + str(value[0]) + " " + str(value[1]))
"""
