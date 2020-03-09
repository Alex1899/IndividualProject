import numpy as np
from scipy.signal import medfilt
from JointAngles import JointAngles
from Functions import find_extremas, filter_extremas, analyse_each_rep, get_evaluation_decision


def evaluate_form(frame_pose, exercise, display):
    if exercise == 'bicep curl':
        return bicep_curl_evaluation(frame_pose, display)
    elif exercise == 'triceps pushdown':
        return triceps_pushdown_evaluation(frame_pose, display)
    elif exercise == 'front raise':
        return front_raise_evaluation(frame_pose, display)
    elif exercise == 'shoulder press':
        return shoulder_press_evaluation(frame_pose, display)
    else:
        print('Error: Wrong exercise.')


def bicep_curl_evaluation(frame_pose, display):
    # check upper arm forearm angle
    # check upper arm trunk angle
    # check trunk knee vector
    # possibly check tempo of the rep (not yet)
    feedback = []
    jointAngles = JointAngles('bicep curl', frame_pose)

    if display is True:
        print('Starting Bicep Curl Analysis...')
        print('Detected side: ' + jointAngles.side)


    upArm_trunk_angles = np.array(jointAngles.upArm_trunk_angles)
    upArm_trunk_angles_filtered = medfilt(medfilt(upArm_trunk_angles, 5), 5)

    upArm_forearm_angles = np.array(jointAngles.upArm_forearm_angles)
    upArm_forearm_angles_filtered = medfilt(medfilt(upArm_forearm_angles, 5), 5)

    trunk_knee_angles = np.array(jointAngles.trunk_knee_angles)
    trunk_knee_angles_filtered = medfilt(medfilt(trunk_knee_angles, 5), 5)

    # Find joint maximass
    extrema = filter_extremas(upArm_forearm_angles_filtered, find_extremas(upArm_forearm_angles_filtered))

    reps_analysis_dict = analyse_each_rep(exercise='bicep curl', string='evaluation', extremas1=extrema,
                                          uf_angles1=upArm_forearm_angles_filtered, ut_angles1=upArm_trunk_angles_filtered,
                                          tk_angles1=trunk_knee_angles_filtered)

    rep_count = len(reps_analysis_dict.keys())

    for key, value in reps_analysis_dict.items():
        # upper arm forearm
        if value['min upper arm forearm'] <= 60:
            feedback1 = 'Good Form: The weight was brought up high enough for a good contraction.\n'

        else:
            # not curling high enough
            if key < int(rep_count*(2/5)):  # 40% of the rep
                feedback1 = 'Bad Form: The weight has not been curled high enough. This could be because the ' \
                            'weight is too heavy.\nFix: Consider lowering the weight to properly target ' \
                            'your biceps and avoid the risk of injury.\n'
            else:
                feedback1 = 'Bad Form:  The weight has not been curled high enough.\nFix: Focus on bringing the ' \
                            'weight higher in order to get maximum biceps contraction.\n'

        if value['max upper arm forearm'] >= 160: # edit threshold
            feedback2 = 'Good Form: The weight was lowered and correct starting position was achieved.\n'
        else:
            feedback2 = 'Bad Form: The weight was lowered half way.\nFix: Lower the weight until you achieve a ' \
                        'correct starting position.\n'

        # upper arm trunk
        if value['max upper arm trunk'] <= 20:
            feedback3 = 'Good Form: Elbows did not move significantly during the movement.\n'
        else:
            feedback3 = 'Bad Form: Elbows have been shifted forward significantly.\nFix: Try to keep your elbows ' \
                        'closer to your body.\n'

        # trunk knee
        if value['min trunk knee'] >= 165:
            feedback4 = 'Good Form: No leaning forward excessively.\n'
        else:
            # leaning forward
            feedback4 = 'Bad Form: Leaning forward significantly.\nFix: Try to keep your back still and straight' \
                        ' throughout the movement.\n'

        if 165 <= value['max trunk knee'] <= 180:
            feedback5 = 'Good Form: No leaning backwards excessively.\n'
        else:
            if value['max trunk knee'] > 180:
                if key < int(rep_count*(2/5)):
                    feedback5 = 'Bad Form: Leaning backwards significantly.This could be because the weight is too heavy.' \
                                ' This puts a lot of pressure on the lower back.\nFix: Consider lowering the weight.' \
                                ' Keep your back straight and focus the effort on the biceps only.\n'
                else:
                    feedback5 = 'Bad Form: Leaning backwards significantly.\nFix Try to keep your back still and straight' \
                                ' throughout the movement.\n'
            else:
                feedback5 = 'Error'

        feedback.append(((feedback1, feedback2, feedback3, feedback4, feedback5), key))

    return get_evaluation_decision(feedback, rep_count, display)


def triceps_pushdown_evaluation(frame_pose, display):
    feedback = []
    jointAngles = JointAngles('triceps pushdown', frame_pose)

    if display is True:
        print('Starting Triceps Pushdown Analysis...')
        print('Detected side: ' + jointAngles.side)

    upArm_trunk_angles = np.array(jointAngles.upArm_trunk_angles)
    upArm_trunk_angles_filtered = medfilt(medfilt(upArm_trunk_angles, 5), 5)

    upArm_forearm_angles = np.array(jointAngles.upArm_forearm_angles)
    upArm_forearm_angles_filtered = medfilt(medfilt(upArm_forearm_angles, 5), 5)

    trunk_knee_angles = np.array(jointAngles.trunk_knee_angles)
    trunk_knee_angles_filtered = medfilt(medfilt(trunk_knee_angles, 5), 5)

    # Find joint maximass
    extrema = filter_extremas(upArm_forearm_angles_filtered, find_extremas(upArm_forearm_angles_filtered, maxima=False), maxima=False)

    reps_analysis_dict = analyse_each_rep(exercise='triceps pushdown', string='evaluation', extremas1=extrema,
                                          uf_angles1=upArm_forearm_angles_filtered, ut_angles1=upArm_trunk_angles_filtered,
                                          tk_angles1=trunk_knee_angles_filtered)

    rep_count = len(reps_analysis_dict.keys())

    for key, value in reps_analysis_dict.items():
        # upper arm forearm
        if value['min upper arm forearm'] >= 63:
            feedback1 = 'Good Form: Correct starting position achieved.\n'

        else:
            feedback1 = 'Bad Form: Incorrect starting position.\nFix: Try not to move your forearms ' \
                        'significantly less than 90 degrees at your elbow joints in order to keep the tension ' \
                        'on the triceps throughout the movement.\n'

        if value['max upper arm forearm'] >= 150:
            feedback2 = 'Good Form: Your arms were fully extended at the bottom part of the move.\n'

        else:
            # not curling high enough
            if key < int(rep_count*(2/5)):  # 40% of the rep
                feedback2 = 'Bad Form: You arms were not fully extended at the bottom part of the move. ' \
                            'This could be because the weight is too heavy.\nFix: Consider lowering the weight ' \
                            'to properly target your triceps and avoid the risk of injury. Focus on fully ' \
                            'extending your arms at the bottom of the move to achieve more exertion ' \
                            'on the triceps.\n'
            else:
                feedback2 = 'Bad Form: You arms were not fully extended at the bottom part of the move.' \
                            '\nFix: Focus on fully extending your arms at the bottom of the move to ' \
                            'achieve more exertion on the triceps.\n'

            # upper arm trunk
        if value['max upper arm trunk'] <= 20:
            feedback3 = 'Good Form: Elbows did not move significantly during the movement.\n'
        else:
            feedback3 = 'Bad Form: Elbows have been shifted forward significantly.\nFix: Try to keep your elbows ' \
                        'closer to your body.\n'

        # trunk knee
        if value['min trunk knee'] >= 150:
            feedback4 = 'Good Form: No leaning forward excessively.\n'
        else:
            if key < int(rep_count*(2/5)):
                # leaning forward
                feedback4 = 'Bad Form: Leaning forward significantly. This could be because the weight is too heavy.' \
                            ' This puts a lot of pressure on the lower back.\nFix: Consider lowering the weight.' \
                            ' Keep your back straight and focus the effort on the triceps only.\n'
            else:
                feedback4 = 'Bad Form: Leaning forward significantly.\nFix: Try to keep your back still and straight' \
                            ' throughout the movement.\n'

        if 150 <= value['max trunk knee'] <= 180:
            feedback5 = 'Good Form: No leaning backwards excessively.\n'
        else:
            feedback5 = 'Bad Form: Leaning backwards significantly.\nFix: Try to keep your back still and straight' \
                        ' throughout the movement.\n'

        feedback.append(((feedback1, feedback2, feedback3, feedback4, feedback5), key))

    return get_evaluation_decision(feedback, rep_count, display)


def shoulder_press_evaluation(frame_pose, display):
    feedback = []
    joint_angles = JointAngles('shoulder press', frame_pose)

    if display is True:
        print('Starting Bicep Curl Analysis...')
        print('Detected side: ' + joint_angles.side)

    left_upArm_trunk_angles = np.array(joint_angles.left_upArm_trunk_angles)
    left_upArm_trunk_angles_filtered = medfilt(medfilt(left_upArm_trunk_angles, 5), 5)

    right_upArm_trunk_angles = np.array(joint_angles.right_upArm_trunk_angles)
    right_upArm_trunk_angles_filtered = medfilt(medfilt(right_upArm_trunk_angles, 5), 5)

    left_upArm_forearm_angles = np.array(joint_angles.left_upArm_forearm_angles)
    left_upArm_forearm_angles_filtered = medfilt(medfilt(left_upArm_forearm_angles, 5), 5)

    right_upArm_forearm_angles = np.array(joint_angles.right_upArm_forearm_angles)
    right_upArm_forearm_angles_filtered = medfilt(medfilt(right_upArm_forearm_angles, 5), 5)

    left_upArm_trunk_minimas = find_extremas(left_upArm_trunk_angles_filtered, maxima=False)
    left_upArm_trunk_minimas = filter_extremas(left_upArm_trunk_angles_filtered, left_upArm_trunk_minimas, False)

    right_upArm_trunk_minimas = find_extremas(right_upArm_trunk_angles_filtered, maxima=False)
    right_upArm_trunk_minimas = filter_extremas(right_upArm_trunk_angles_filtered, right_upArm_trunk_minimas, False)

    reps_analysis_dict = analyse_each_rep(exercise='shoulder press', string='evaluation',
                                          extremas1=left_upArm_trunk_minimas, uf_angles1=left_upArm_forearm_angles_filtered,
                                          ut_angles1=left_upArm_trunk_angles_filtered, extremas2=right_upArm_trunk_minimas,
                                          uf_angles2=right_upArm_forearm_angles_filtered, ut_angles2=right_upArm_trunk_angles_filtered)

    rep_count = len(reps_analysis_dict.keys())

    for key, value in reps_analysis_dict.items():
        # min forearms and upper arms
        if 87 >= value[0]['min left upper arm forearm'] >= 55 and 87 >= value[1]['min right upper arm forearm'] >= 55:
            if max(value[0]['min left upper arm forearm'], value[1]['min right upper arm forearm']) - min(value[0]['min left upper arm forearm'], value[1]['min right upper arm forearm']) > 18:
                # detect which one is at max angle
                max_angle = max(value[0]['min left upper arm forearm'], value[1]['min right upper arm forearm'])

                if value[0]['min left upper arm forearm'] == max_angle:
                    feedback1 = 'Bad Form: Your right forearm is moved towards your right shoulder significantly.\nFix: Focus on having ' \
                                'your forearms at equal distances from your shoulders in order to avoid muscular imbalances.\n'
                else:
                    feedback1 = 'Bad Form: Your left forearm is moved towards yourleft shoulder significantly.\nFix: Focus on having ' \
                                'your forearms at equal distances from your shoulders in order to avoid muscular imbalances.\n'
            else:
                feedback1 = 'Good Form: Your forearms are almost perpendicular to your upper arms.\n'

        else:
            if value[0]['min left upper arm forearm'] < 55 and value[1]['min right upper arm forearm'] < 55:
                feedback1 = 'Bad Form: Incorrect starting position. Your forearms showed significant movement ' \
                            'towards your shoulders.\nFix: Try to not move your forearms significantly less than 90' \
                            'degrees at your elbow joints.\n'

            elif value[0]['min left upper arm forearm'] < 55:
                feedback1 = 'Bad Form: Incorrect starting position. Your left forearm showed significant movement ' \
                            'towards your left shoulder.\nFix: Try to not move your left forearm significantly less than 90'\
                            'degrees at your left elbow joint.\n'

            elif value[1]['min right upper arm forearm'] < 55:
                feedback1 = 'Bad Form: Incorrect starting position. Your right forearm showed significant movement ' \
                            'towards your left shoulder.\nFix: Try to not move your right forearm significantly less than 90' \
                            'degrees at your right elbow joint.\n'

            elif value[0]['min left upper arm forearm'] > 87 and value[1]['min right upper arm forearm'] > 87:
                feedback1 = 'Bad Form: Incorrect starting position. Your forearms are at the incorrect level to your upper arms.' \
                            '\nFix: Try to have your forearms almost perpendicular to your upper arms.\n'

            elif value[0]['min left upper arm forearm'] > 87:
                feedback1 = 'Bad Form: Incorrect starting position. Your left forearm is at the incorrect level to your left upper arm.' \
                            '\nFix: Try to have your left forearm almost perpendicular to your left upper arm.\n'

            elif value[1]['min right upper arm forearm'] > 87:
                feedback1 = 'Bad Form: Incorrect starting position. Your right forearm are at the incorrect level to your right upper arm.' \
                            '\nFix: Try to have your right forearm almost perpendicular to your right upper arm.\n'
            else:
                feedback1 = 'Error'

        # min upper arms and trunk
        if 62 <= value[0]['min left upper arm trunk'] <= 99 and 62 <= value[1]['min right upper arm trunk'] <= 99:
            if max(value[0]['min left upper arm trunk'], value[1]['min right upper arm trunk']) - min(value[0]['min left upper arm trunk'], value[1]['min right upper arm trunk']) > 15:
                # detect which one is at max angle
                max_angle = max(value[0]['min left upper arm trunk'], value[1]['min right upper arm trunk'])

                if value[0]['min left upper arm trunk'] == max_angle:
                    feedback2 = 'Bad Form: Your right elbow is moved moved down significantly compared to your left.\nFix: Focus on having ' \
                                'your elbows at equal angles to your torso in order to avoid muscular imbalances.\n'
                else:
                    feedback2 = 'Bad Form: Your left elbow is moved down significantly compared to your right.\nFix: Focus on having ' \
                                'your elbows at equal angles to your torso in order to avoid muscular imbalances.\n'
            else:
                feedback2 = 'Good Form: Your upper arms are almost perpendicular to the torso. Correct starting position achieved.\n'

        else:
            if value[0]['min left upper arm trunk'] < 62 and value[1]['min right upper arm trunk'] < 62:
                feedback2 = 'Bad Form: Incorrect starting position. Your elbows were at incorrect angles to your torso.\nFix: Focus on having' \
                            ' your elbows almost perpendicular to your torso.\n'

            elif value[0]['min left upper arm trunk'] < 62:
                feedback2 = 'Bad Form: Incorrect starting position. Your left elbows was at incorrect andle to your torso.\nFix: Focus on having' \
                            ' your left elbow almost perpendicular to your torso.\n'

            elif value[1]['min right upper arm trunk'] < 62:
                feedback2 = 'Bad Form: Incorrect starting position. Your right elbow was at incorrect andle to your torso.\nFix: Focus on having' \
                            ' your right elbow almost perpendicular to your torso.\n'

            elif value[0]['min left upper arm trunk'] > 99 and value[1]['min right upper arm trunk'] > 99:
                feedback2 = 'Bad Form: Your upper arms are at incorrect angles to your torso.\nFix: Try to have your upper arms' \
                            ' almost perpendicular to your torso.\n'

            elif value[0]['min left upper arm trunk'] > 99:
                feedback2 = 'Bad Form: Your left upper arm is at the incorrect angle to your torso.\nFix: Try to have your left upper arm' \
                            ' almost perpendicular to your torso.\n'

            elif value[1]['min right upper arm trunk'] > 99:
                feedback2 = 'Bad Form: Your right upper arm is at the incorrect angle to your torso.\nFix: Try to have your right upper arm' \
                            ' almost perpendicular to your torso.\n'
            else:
                feedback2 = 'Error'

        """
        # max upper arm trunk
        if 180 >= value[0]['max left upper arm trunk'] >= 136 and 180 >= value[1]['max right upper arm trunk'] >= 136:
            feedback3 = 'Good Form: Your elbows were brought up high enough.\n'

        elif value[0]['max left upper arm trunk'] < 136 and value[1]['max right upper arm trunk'] < 136:
            feedback3 = 'Bad Form: Your elbows were not brought up high enough for a good shoulders contraction.\nFix:' \
                        ' Focus on bringing your elbows up to engage your shoulders more throughout the movement.\n'

        elif value[0]['max left upper arm trunk'] < 136:
            feedback3 = 'Bad Form: You left elbow was not brought up high enough for a good shoulder contraction.' \
                        '\nFix: Focus on bringing your left elbow up to engage your shoulder more throughout the movement.\n'

        elif value[1]['max right upper arm trunk'] < 136:
            feedback3 = 'Bad Form: You right elbow was not brought up high enough for a good shoulder contraction.' \
                        '\nFix: Focus on bringing your right elbow up to engage your shoulder more throughout the movement.\n'

        elif value[0]['max left upper arm trunk'] > 180 and value[1]['max right upper arm trunk'] > 180:
            feedback3 = 'Bad Form: Your elbows are at the incorrect angles to your torso.\nFix: Try not to cross your arms' \
                        ' excessively at the top part of the movement.\n'

        elif value[0]['max left upper arm trunk'] > 180:
            feedback3 = 'Bad Form: Your left elbow is at the incorrect angle to your torso.\nFix: Try not to move your left arm' \
                        ' to the right excessively at the top part of the movement.\n'

        elif value[1]['max right upper arm trunk'] > 180:
            feedback3 = 'Bad Form: Your right elbow is at the incorrect angle to your torso.\nFix: Try not to move your right arm' \
                        ' to the left excessively at the top part of the movement.\n'
        else:
            feedback3 = 'Error'
        """

        # all parts
        if 131 <= value[0]['max left upper arm forearm'] <= 172 and 131 <= value[1]['max right upper arm forearm'] <= 172\
                and 136 <= value[0]['max left upper arm trunk'] <= 174 and 136 <= value[1]['max right upper arm trunk'] <= 174:
            if max(value[0]['max left upper arm forearm'], value[1]['max right upper arm forearm']) - min(value[0]['max left upper arm forearm'], value[1]['max right upper arm forearm']) > 15:
                # detect which one is at max angle
                max_angle = max(value[0]['max left upper arm forearm'], value[1]['max right upper arm forearm'])

                if value[0]['max left upper arm forearm'] == max_angle:
                    feedback4 = 'Bad Form: Your right arm is not fully extended.\nFix: Focus on extending your arms ' \
                                'equally in order to avoid muscular imbalances.\n'
                else:
                    feedback4 = 'Bad Form: Your left arm is not fully extended.\nFix: Focus on extending your arms ' \
                                'equally in order to avoid muscular imbalances.\n'

            else:
                feedback4 = 'Good Form: The weight was brought up high enough.\n'

        else:
            if value[0]['max left upper arm forearm'] < 131 and value[1]['max right upper arm forearm'] < 131\
                    and value[0]['max left upper arm trunk'] < 136 and value[1]['max right upper arm trunk'] < 136:
                if key < int(rep_count*(2/5)):  # 40% of the total reps
                    feedback4 = 'Bad Form: The weight was not brought up high enough. This could be because the weight is too heavy.' \
                                '\nFix: Consider lowering the weight to target you shoulders better and avoid the ristk of injury.\n'
                else:
                    feedback4 = 'Bad Form: The weight was not brought up high enough.\nFix: Try not to perform partial reps and ' \
                                'focus on bringing the weight up until your elbows extend fully.\n'

            elif value[0]['max left upper arm forearm'] < 131 and value[1]['max right upper arm forearm'] < 131:
                feedback4 = 'Bad Form: Your elbow joints were not extended fully.\nFix: Focus on extending your ' \
                            'elbows at the top part of the move to engage your shoulders more.\n'

            elif value[0]['max left upper arm forearm'] < 131:
                feedback4 = 'Bad Form: Your left elbow joint were not extended fully.\nFix: Focus on extending your ' \
                            'left elbow at the top part of the move to engage your shoulder more.\n'

            elif value[1]['max right upper arm forearm'] < 131:
                feedback4 = 'Bad Form: Your right elbow joint were not extended fully.\nFix: Focus on extending your ' \
                            'right elbow at the top part of the move to engage your shoulder more.\n'
            else:
                feedback4 = 'Error'

        feedback.append(((feedback1, feedback2, feedback4), key))

    return get_evaluation_decision(feedback, rep_count, display)


def front_raise_evaluation(frame_pose, display):
    joint_angles = JointAngles('front raise', frame_pose)
    feedback = []

    if display is True:
        print('Starting Front Raise Analysis...')
        print('Detected side: ' + joint_angles.side)

    upArm_trunk_angles = np.array(joint_angles.upArm_trunk_angles)
    upArm_trunk_angles_filtered = medfilt(medfilt(upArm_trunk_angles, 5), 5)

    upArm_forearm_angles = np.array(joint_angles.upArm_forearm_angles)
    upArm_forearm_angles_filtered = medfilt(medfilt(upArm_forearm_angles, 5), 5)

    trunk_knee_angles = np.array(joint_angles.trunk_knee_angles)
    trunk_knee_angles_filtered = medfilt(medfilt(trunk_knee_angles, 5), 5)

    upArm_trunk_minimas = find_extremas(upArm_trunk_angles_filtered, maxima=False)
    upArm_trunk_minimas = filter_extremas(upArm_trunk_angles_filtered, upArm_trunk_minimas, maxima=False)

    reps_analysis_dict = analyse_each_rep(exercise='front raise', string='evaluation', extremas1=upArm_trunk_minimas,
                                          uf_angles1=upArm_forearm_angles_filtered, ut_angles1=upArm_trunk_angles_filtered,
                                          tk_angles1=trunk_knee_angles_filtered)

    rep_count = len(reps_analysis_dict.keys())

    for key, value in reps_analysis_dict.items():
        if 180 >= value['min upper arm forearm'] >= 148 and 155 <= value['max upper arm forearm'] <= 180:
            feedback1 = 'Good Form: Your forearm is at the correct angle to you upper arm.\n'
        else:
            feedback1 = 'Bad Form: Your elbow is not extended enough.\nFix: Focus on having your elbows extended' \
                        ' throughout the movement for a better front deltoids contraction.\n'

        if 13 >= value['min upper arm trunk'] >= 0.02:
            feedback2 = 'Good Form: The weight was lowered fully and correct starting position achieved.\n'

        else:
            feedback2 = 'Bad Form: Incorrect starting position. The weight was lowered halfway.\nFix: Focus on lowering' \
                        ' the weight fully at the bottom part of the move to achieve a correct starting position.\n'
        # theres me in one video with 121 angle of ut
        if 69 <= value['max upper arm trunk'] <= 122:
            feedback3 = 'Good Form: The weight was brought up high enough. Correct finishing position.'

        elif value['max upper arm trunk'] < 69:
            if key < int(rep_count * (2/5)):
                feedback3 = 'Bad Form: The weight has not been brought up high enough. This could be because the weight' \
                            ' is too heavy.\nFix: Consider lowering the weight to properly target your front deltoids ' \
                            'and avoid the risk of injury.\n'
            else:
                feedback3 = 'Bad Form: The weight has not been brought up high enough.\nFix: Focus on bringing the weight' \
                            'higher than your shoulder level to properly target your front deltoids.\n'

        elif value['max upper arm trunk'] > 122:
            feedback3 = 'Bad Form: The weight has been brought up way too much.\nFix: Try not to bring the weight ' \
                        'higher than your shoulder level in order to keep the tension on front deltoids.\n'
        else:
            feedback3 = 'Error'

        if value['min trunk knee'] >= 154:
            feedback4 = 'Good Form: No leaning forward excessively.\n'

        else:
            feedback4 = 'Bad Form: Leaning forward significantly.\nFix: Try to keep your back still and straight' \
                       ' throughout the movement.\n'

        if 154 <= value['max trunk knee'] <= 180:
            feedback5 = 'Good Form: No leaning backwards excessively.\n'

        elif value['max trunk knee'] > 180:
            if key < int(rep_count*(2/5)):
                feedback5 = 'Bad Form: Leaning backwards significantly. This could be because the weight is too heavy.' \
                            ' This puts a lot of pressure on the lower back.\nFix: Consider lowering the weight.' \
                            ' Keep your back straight and focus the effort on front deltoids only.\n'
            else:
                feedback5 = 'Bad Form: Leaning backwards significantly.\nFix: Try to keep your back still and straight' \
                            ' throughout the movement.\n'
        else:
            feedback5 = 'Error'

        feedback.append(((feedback1, feedback2, feedback3, feedback4, feedback5), key))

    return get_evaluation_decision(feedback, rep_count, display)

