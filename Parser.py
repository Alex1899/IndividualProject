import glob
import json
import os
import numpy as np
from Frame import FramePose

input_folder = "C:\\Users\\AK5U16\\Downloads\\openpose\\bicepvid"


def parse_frames(path_to_jsons):
    """ Gets keypoints from each frame and passes them to FramePose class.
     Args:
         path_to_jsons: path to the folder where json files extracted using OpenPose are located.
     return:
         frame_poses: a list containing FramePose objects where all keypoints for each frame are saved
     """
    json_files = glob.glob(os.path.join(path_to_jsons, "*.json"))
    num_json_files = len(json_files)
    frame_poses = []
    for num in range(num_json_files):
        with open(json_files[num]) as obj:
            file_json = json.load(obj)
            keypoints = np.array(file_json['people'][0]['pose_keypoints_2d'])
            pose = FramePose(keypoints.reshape((25, 3)))
            frame_poses.append(pose)
    print(len(frame_poses))

    return frame_poses

