import argparse
import os
import glob
import subprocess
from Parser import parse_frames
from Evaluation import evaluate_form


def main():
    parser = argparse.ArgumentParser(description='Exercise Form Evaluation')
    parser.add_argument('--mode', type=str, default='evaluation')
    parser.add_argument('--exercise', type=str, help='name of the exercise to evaluate ex. "bicep curl"')
    parser.add_argument('--video_path', type=str, help='path to video to evaluate')
    parser.add_argument('--videos_folder', type=str, default='videos', help='folder where all exercise videos are stored')
    parser.add_argument('--keypoints_folder', type=str, default='keypoints_for_all', help='all keypoints folder')
    parser.add_argument('--output_videos_folder', type=str, default='output_videos', help='output video folder in .avi')

    arguments = parser.parse_args()

    # extract keypoints from all videos in videos folder and store in keypoints_for_all folder.
    if arguments.mode == 'keypoints_extraction':
        os.chdir('../openpose')
        print(os.getcwd())

        if arguments.exercise:
            video_folders = glob.glob(os.path.join(os.getcwd(), arguments.videos_folder, arguments.exercise))
        else:
            video_folders = glob.glob(os.path.join(os.getcwd(), arguments.videos_folder + '/*'))
        print(video_folders)

        os.chdir('../IndividualProject')
        print(os.getcwd())
        for vid_folder in video_folders:
            exercise_videos = glob.glob(vid_folder + '/*')
            #print(exercise_videos)

            for video in exercise_videos:
                video_name = os.path.basename(video)
                points_folder_name = str(video_name.split('.', 1)[0])
                output_points_folder = os.path.join(arguments.keypoints_folder, vid_folder, str(points_folder_name))
                print(output_points_folder)

                if not os.path.exists(output_points_folder):
                    os.makedirs(output_points_folder)

                os.chdir('../openpose')
                print(os.getcwd())
                print(arguments.output_videos_folder)
                print(vid_folder)
                output_videos_folder = os.path.join(os.getcwd(), arguments.output_videos_folder) + '/' + vid_folder
                print(output_points_folder)
                if not os.path.exists(output_videos_folder):
                    os.makedirs(output_videos_folder)
                #print(output_videos_folder)
                break
                output_video = output_points_folder + '/' + points_folder_name + '.avi'
                openpose_demo = os.path.join('bin', 'OpenPoseDemo.exe')
                """
                subprocess.call([openpose_demo, '--video', video, '--write_json', output_points_folder, '--write_video',
                                    output_video, '--number_people_max', '1'])
                """

    # if a specific video path is specified, evaluate that video and save keypoints/output video in the folder
    elif arguments.mode == 'evaluation':
        if arguments.video_path and arguments.exercise:
            video_name = str(os.path.basename(arguments.video_path).split('.', 1)[0])
            output_points_folder = os.path.join(arguments.keypoints_folder, arguments.exercise, video_name)
            print(output_points_folder)

            if not os.path.exists(output_points_folder):
                os.makedirs(output_points_folder)

            os.chdir('../openpose')
            output_videos_folder = os.path.join(arguments.output_videos_folder, arguments.exercise)
            if not os.path.exists(output_videos_folder):
                os.makedirs(output_videos_folder)

            print(output_videos_folder)
            output_video = output_points_folder + '/' + video_name + '.avi'
            openpose_demo = os.path.join('bin', 'OpenPoseDemo.exe')
            """
            subprocess.call([openpose_demo, '--video', arguments.video_path, '--write_json', output_points_folder, '--write_video',
                             output_video, '--number_people_max', '1'])
            """

            frame_pose = parse_frames(output_points_folder)
            evaluate_form(frame_pose, arguments.exercise, True)

        else:
            print('Please specify the video path and exercise to evaluate.')

    else:
        print('Wrong mode. Please use one of the followings: evaluation, keypoints_extraction.')


if __name__ == "__main__":
    main()





