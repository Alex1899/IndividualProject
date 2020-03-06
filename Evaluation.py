import numpy as np
from scipy.signal import medfilt
from JointAngles import JointAngles
from Functions import find_extremas, filter_extremas, analyse_each_rep

"""
def evaluate_form(frame_pose, exercise):
    if exercise == 'bicep curl'
        return bicep_curl_evaluation(frame_pose)
"""


def bicep_curl_evaluation(frame_pose):
    # check upper arm forearm angle
    # check upper arm trunk angle
    # check trunk knee vector
    # possibly check tempo of the rep (not yet)

    jointAngles = JointAngles('bicep curl', frame_pose)
    print('Starting Bicep Curl Analysis...')
    print('Detected arm: ' + jointAngles.side)

    upArm_trunk_angles = np.array(jointAngles.upArm_trunk_angles)
    upArm_trunk_angles_filtered = medfilt(medfilt(upArm_trunk_angles, 5), 5)

    upArm_forearm_angles = np.array(jointAngles.upArm_forearm_angles)
    upArm_forearm_angles_filtered = medfilt(medfilt(upArm_forearm_angles, 5), 5)

    trunk_knee_angles = np.array(jointAngles.trunk_knee_angles)
    trunk_knee_angles_filtered = medfilt(medfilt(trunk_knee_angles, 5), 5)

    # Find upper arm and trunk maximum angles
    upArm_trunk_maximas = filter_extremas(find_extremas(upArm_trunk_angles_filtered))

    # Find trunk and knee maximum angles
    trunk_knee_maximas = filter_extremas(find_extremas(trunk_knee_angles_filtered))

    # Count repetitions
    upArm_forearm_maximas = filter_extremas(find_extremas(upArm_forearm_angles_filtered))

    # Find upper arm and forearm minimum points to count reps
    upArm_forearm_minimas = filter_extremas(find_extremas(upArm_forearm_angles_filtered, maxima=False), maxima=False)




