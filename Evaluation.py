import numpy as np
from scipy.signal import medfilt
from JointAngles import JointAngles
from Functions import find_extremas, filter_extremas, analyse_each_rep

"""
def evaluate_form(frame_pose, exercise):
    if exercise == 'bicep curl'
        return bicep_curl_evaluation(frame_pose)
"""


def bicep_curl_and_triceps_pushdown_evaluation(exercise, frame_pose):
    # check upper arm forearm angle
    # check upper arm trunk angle
    # check trunk knee vector
    # possibly check tempo of the rep (not yet)

    text1 = 'The weight has not been curled high enough.'
    text2 = '\nFix: Try to keep your back still and straight throughout the movement.'
    text3 = 'This could be because the weight is too heavy. This puts a lot of pressure on the lower back.' \
            '\nFix: Consider lowering the weight. Keep your back straight and focus the effort on the biceps only. ' \

    min_uf_threshold = 60
    max_uf_threshold = 160
    tk_threshold = 165
    feedback = []


    jointAngles = JointAngles('bicep curl', frame_pose)
    print('Starting Bicep Curl Analysis...')
    print('Detected arm: ' + jointAngles.side)

    upArm_trunk_angles = np.array(jointAngles.upArm_trunk_angles)
    upArm_trunk_angles_filtered = medfilt(medfilt(upArm_trunk_angles, 5), 5)

    upArm_forearm_angles = np.array(jointAngles.upArm_forearm_angles)
    upArm_forearm_angles_filtered = medfilt(medfilt(upArm_forearm_angles, 5), 5)

    trunk_knee_angles = np.array(jointAngles.trunk_knee_angles)
    trunk_knee_angles_filtered = medfilt(medfilt(trunk_knee_angles, 5), 5)

    # Find joint maximass
    if exercise == 'bicep curl':
        extremas = filter_extremas(find_extremas(upArm_forearm_angles_filtered))

    elif exercise == 'triceps pushdown':
        extrema = filter_extremas(find_extremas(upArm_trunk_angles_filtered), maxima=False)
        text1 = 'The weight has not been pushed down enough.'
        text2 = 'This could be because the weight is too heavy. This puts a lot of pressure on the lower back.' \
                '\nFix: Try to not lean forward excessively throughout the movement.'
        text3 = '\nFix: Try to keep your back still and straight throughout the movement.'

        min_uf_threshold = 63
        max_uf_threshold = 175
        tk_threshold = 150

    reps_analysis_dict = analyse_each_rep(exercise='bicep curl', string='evaluation', extremas1=extrema,
                                          uf_angles1=upArm_forearm_angles_filtered, ut_angles1=upArm_trunk_angles_filtered,
                                          tk_angles1=trunk_knee_angles_filtered)

    rep_count = len(reps_analysis_dict.keys())

    for key, value in reps_analysis_dict.items():
        # upper arm forearm
        if value['min upper arm forearm'] <= min_uf_threshold:
            feedback1 = 'Good Form: Repetition performed with a good technique.\n'
        else:
            # not curling high enough
            if key < 4:
                feedback1 = 'Bad Form: ' + text1 + 'This could be because the ' \
                            'weight is too heavy.\nFix: Consider lowering the weight to properly target your ' \
                            'biceps and avoid the risk of injury.\n'
            else:
                feedback1 = 'Bad Form:' + text1 + '\nFix: Focus on bringing the ' \
                            'weight higher in order to get maximum biceps contraction.\n'

        if value['max upper arm forearm'] >= max_uf_threshold:
            feedback2 = 'Good Form: Repetition performed with a good technique.\n'
        else:
            feedback2 = 'Bad Form: The weight was lowered half way.\nFix: Extend your arms fully on the way down.'

        # upper arm trunk
        if value['max upper arm trunk'] <= 20:
            feedback3 = 'Good Form: Repetition performed with a good technique.\n'
        else:
            feedback3 = 'Bad Form: Elbows have been shifted forward significantly.\nFix: Try to keep your elbows ' \
                        'closer to your body.'

        # trunk knee
        if value['min trunk knee'] >= tk_threshold:
            feedback4 = 'Good Form: Repetition performed with a good technique.\n'
        else:
            # leaning forward
            feedback4 = 'Feedback: Leaning forward significantly.' + text2

        if tk_threshold <= value['max trunk knee'] <= 180:
            feedback5 = 'Feedback: Repetition performed with a good technique.\n'
        else:
            feedback5 = 'Feedback: Leaning backwards significantly.' + text3

        feedback.extend((feedback1, feedback2, feedback3, feedback4, feedback5))

    if 'Good' in feedback:
        return feedback


























