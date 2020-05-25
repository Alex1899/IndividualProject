import os
import sys

hm_files_path = os.path.join(os.getcwd(), 'Heuristic_model')

if hm_files_path in sys.path:
    pass
else:
    sys.path.append(hm_files_path)
import Parser as pr
import Evaluation as eval
from Frame import FramePose
from JointAngles import JointAngles
from Functions import find_extremas
import argparse
import glob
import subprocess
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import json
import numpy as np

exercise = ''
feedback = []
json_files_list = []
upArm_forearm_angles = np.array([])
upArm_trunk_angles = np.array([])
trunk_knee_angles = np.array([])
rep_count = 0
p = None


class JSONHandler(FileSystemEventHandler):

    def on_created(self, event):
        #print(event.src_path)
        global exercise

        # json_files_list.append(event.src_path)
        # print(event.src_path)
        if exercise == 'bicep_curl':
            biceps_real_time_analysiss(event.src_path)


def main():
    parser = argparse.ArgumentParser(description='Exercise Form Evaluation')
    parser.add_argument('--mode', type=str, default='evaluation')
    parser.add_argument('--exercise', type=str, help='name of the exercise to evaluate ex. bicep_curl')
    # parser.add_argument('--video_path', type=str, help='path to video to evaluate')
    # parser.add_argument('--videos_folder', type=str, default='videos',
    #                   help='folder where all exercise videos are stored')
    parser.add_argument('--keypoints_folder', type=str, default='real_time_keypoints',
                        help='real time video keypoints folder')
    parser.add_argument('--output_videos_folder', type=str, default='output_videos', help='output video folder in .avi')

    arguments = parser.parse_args()

    if arguments.mode == 'evaluation':
        if arguments.exercise:
            global exercise
            exercise = arguments.exercise
            os.chdir('../IndividualProject')
            dateTime = datetime.now()
            time2 = dateTime.strftime("%d-%b-%Y-(%H-%M-%S)")
            video_name = arguments.exercise + '_' + time2

            output_points_folder = os.path.join(os.getcwd(), arguments.keypoints_folder, exercise, video_name)

            # if keypoints already extracted then skip extraction process
            if not os.path.exists(output_points_folder):
                print('Starting the video evaluation...')
                print("Running OpenPose on the input video...")
                print("Video name: " + str(os.path.basename(video_name)))
                os.makedirs(output_points_folder)
                os.chdir('../openpose')
                output_videos_folder = os.path.join(os.getcwd(), arguments.output_videos_folder, exercise,
                                                    video_name)

                if not os.path.exists(output_videos_folder):
                    os.makedirs(output_videos_folder)
                    output_video = os.path.join(output_videos_folder, video_name + '.avi')
                    ls = ['--write_video', output_video]
                else:
                    ls = []

                openpose_demo = os.path.join('bin', 'OpenPoseDemo.exe')
                observer = Observer()
                event_handler = JSONHandler()  # create event handler
                # set observer to use created handler in directory
                observer.schedule(event_handler, path=output_points_folder)
                observer.start()
                global p
                p = subprocess.Popen([openpose_demo, '--write_json', output_points_folder,
                                      '--number_people_max', '1'] + ls)

                # sleep until keyboard interrupt, then stop + rejoin the observer
                try:
                    while True:
                        time.sleep(.200)
                except KeyboardInterrupt:
                    observer.stop()

                observer.join()

            else:
                print("Processing the video...")

        else:
            print('Please specify the video path and exercise to evaluate.')

    else:
        print('Wrong mode. Please use one of the followings: evaluation, keypoints_extraction.')


def biceps_real_time_analysiss(js_file):
    global upArm_forearm_angles
    global upArm_trunk_angles
    global trunk_knee_angles
    global rep_count

    frame_pose_obj = pr.parse_frames(js_file)
    if frame_pose_obj is None:
        print('--'*20 + '- ERROR -' + '--'*20)
        print('Error: Could not detect the pose. Please, make sure your whole body is in the frame.')
        print('\n')
        return

    jointAngles = JointAngles('bicep_curl', frame_pose_obj)
    if not jointAngles.parts_filtered:
        print('--' * 20 + '- ERROR -' + '--' * 20)
        print('Error: Could not detect all body parts. Please, make sure all of your body parts are in the frame.')
        print('\n')
        return

    upArm_forearm_angles = np.append(upArm_forearm_angles, jointAngles.upArm_forearm_angles[0])
    upArm_trunk_angles = np.append(upArm_trunk_angles, jointAngles.upArm_trunk_angles[0])
    trunk_knee_angles = np.append(trunk_knee_angles, jointAngles.trunk_knee_angles[0])

    extrema = find_extremas(np.array(upArm_forearm_angles))

    if extrema.size == 0:
        pass
    else:
        rep_count += 1
        upArm_forearm_angles, upArm_trunk_angles, trunk_knee_angles = eval.bicep_curl_evaluation(extrema,
                                                                                                 np.array(
                                                                                                     upArm_forearm_angles),
                                                                                                 np.array(
                                                                                                     upArm_trunk_angles),
                                                                                                 np.array(
                                                                                                     trunk_knee_angles),
                                                                                                 rep_count)



if __name__ == "__main__":
    main()
