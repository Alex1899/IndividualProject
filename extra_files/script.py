import os
import glob
import subprocess
import pathlib

videos_folder = "C:\\Users\\ak5u16\\Desktop\\openpose\\videos"
exercise_folders = glob.glob(videos_folder + '/*/')
openpose_path = os.getcwd()

print('Openpose path: ' + openpose_path)
exercise_videos = []
for video_folder in exercise_folders:
    exercise_videos += glob.glob(video_folder + '/*')

for video in exercise_videos:
    if 'triceps' in video or 'bicep' in video:
        continue
    video_name = os.path.basename(video)
    points_folder_name = video_name.split('.', 1)[0]
    folder = video_name.split('_', 1)[0]
    print(points_folder_name)

    output_points_folder = glob.glob(os.path.join(openpose_path, 'keypoints_for_all/' + str(folder) + '*/'
                                             + '*' + str(points_folder_name)))
    output_videos_folder = 'output_videos/' + str(points_folder_name) + '.avi'
    print(output_videos_folder)

    openpose_demo = os.path.join('bin', 'OpenPoseDemo.exe')
    subprocess.call([openpose_demo, '--video', video, '--write_json', output_points_folder, '--write_video', output_videos_folder, '--number_people_max', '1'])








