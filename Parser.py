import argparse, glob, json, os
import numpy as np


def main():
    input_folder = "C:\\Users\\AK5U16\\Downloads\\openpose\\bicepvid"
    output_folder = "C:\\Users\\AK5U16\\Downloads\\openpose\\bicepvid_output"
    paths_json_frames = glob.glob(os.path.join(input_folder, '*.json'))
    paths_json_frames = sorted(paths_json_frames)

    # Get all json files for each video
    all_frames = [parse_frames(input_folder, output_folder)]
    return all_frames


def parse_frames(path_to_jsons, output_folder):
    json_files = glob.glob(os.path.join(path_to_jsons, "*.json"))
    num_json_files = len(json_files)
    print(num_json_files)
    all_points = np.zeros((num_json_files, 25, 3))
    for num in range(num_json_files):
        with open(json_files[num]) as obj:
            file_json = json.load(obj)
            keypoints = np.array(file_json['people'][0]['pose_keypoints_2d'])
            all_points[num] = keypoints.reshape((25, 3))

    print(path_to_jsons)
    print(all_points[0])
    output = os.path.join(output_folder, os.path.basename(path_to_jsons))
    np.save(output, all_points)



if __name__ == '__main__':
    main()