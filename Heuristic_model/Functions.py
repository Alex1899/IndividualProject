import sys
import subprocess

"""
try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "--user"])
"""
import numpy as np


# Detecting sides (left or right) on videos
# can use this method for other exercises
def detect_side(frame_poses):
    right_joints = ['RSHOULDER', 'RELBOW', 'RWRIST', 'RHIP', 'RKNEE', 'REYE', 'REAR']
    right_parts = [[posture.joint_keypoints[joint] for joint in right_joints] for posture in frame_poses]
    right_parts_filtered = [1 for part in right_parts if all(joint_points[2] != 0 for joint_points in part)]

    left_joints = ['LSHOULDER', 'LELBOW', 'LWRIST', 'LHIP', 'LKNEE', 'LEYE', 'LEAR']
    left_parts = [[posture.joint_keypoints[joint] for joint in left_joints] for posture in frame_poses]
    left_parts_filtered = [1 for part in left_parts if all(joint_points[2] != 0 for joint_points in part)]

    right_sum = sum(right_parts_filtered)
    left_sum = sum(left_parts_filtered)
    # print('Right side: ' + str(right_sum))
    # print('Left  side: ' + str(left_sum))

    # think about the case when they are equal
    if right_sum > left_sum:
        side = 'right'
    else:
        side = 'left'

    return side


# Finds extrema points in numpy arrray
def find_extremas(filtered_nparray, maxima=True):
    if maxima:
        points = _boolrelextrema(filtered_nparray, np.greater, order=1)
    else:
        points = _boolrelextrema(filtered_nparray, np.less, order=1)

    indexes = np.nonzero(points)[0]
    extrema_points = np.take(filtered_nparray, indexes)

    return extrema_points


# Method used for finding local minima points for repetition counting
# https://github.com/scipy/scipy/blob/master/scipy/signal/_peak_finding.py
def _boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    if (int(order) != order) or (order < 1):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)
    data = np.ma.masked_array(data, mask=np.hstack(([1], np.diff(data))) == 0)
    if np.ma.is_masked(data):
        locs = locs[np.ma.getmask(data) == False]
        main = data.take(locs, axis=axis, mode=mode)
        results = np.zeros(data.shape, dtype=bool)
        for index, result in enumerate(_boolrelextrema(main, comparator, axis=axis, order=order, mode=mode)):
            results[locs[index]] = result
        return results
    else:
        locs = locs[np.ma.getmask(data) == False]
        results = np.ones(data.shape, dtype=bool)
        main = data.take(locs, axis=axis, mode=mode)
        for shift in range(1, order + 1):
            plus = data.take(locs + shift, axis=axis, mode=mode)
            minus = data.take(locs - shift, axis=axis, mode=mode)
            results &= comparator(main, plus)
            results &= comparator(main, minus)
            if ~results.any():
                return results
        return results


# Count frames between two extrema points
def count_angles_between_two_points(extrema1, extrema2, angles_array):
    index1 = np.argwhere(angles_array == extrema1)[0][0]
    index2 = np.argwhere(angles_array == extrema2)[0][0]

    nums = np.arange(min(index1, index2) + 1, max(index1, index2))
    # pm = np.take(angles_array, nums)
    count = nums.size

    return count


def count_angles_between_extremas(angles_array, extremas_array):
    count1 = 0
    # do not count points before first extrema point
    indexes = np.argwhere(angles_array == extremas_array[0])
    angles_array = np.delete(angles_array, np.arange(indexes[0][0]))

    extremas = extremas_array
    # calculate the average count value

    count_list = []
    for x_point in angles_array:
        if x_point not in extremas:
            count1 += 1
        else:
            extremas = extremas[1:]

            if count1 == 0:
                continue
            else:
                count_list.append(count1)
                count1 = 0

    return count_list


def filter_extrema_by_angles_number_inbetween(extremas_array, count_list, ls, maxima=True):
    points_to_delete = []
    counts_to_remove = []
    ls_copy = ls
    size = len(ls)
    # print('counts less than threshold: ' + str(ls))
    if size > 0:
        while len(ls) != 0:
            for n in ls:
                # print('count: ' + str(n))
                idx = count_list.index(n)
                point1 = extremas_array[idx]
                point2 = extremas_array[idx + 1]
                min_point = min(point1, point2)
                max_point = max(point1, point2)

                if maxima:
                    point_to_remove = min_point
                    start_point = max_point
                else:
                    point_to_remove = max_point
                    start_point = min_point

                # print('point1: ' + str(point1) + ' & point2: ' + str(point2))
                # print('deleted: ' + str(point_to_remove))

                # store points and indexes to remove
                points_to_delete.append(point_to_remove)
                point_to_remove_index = np.argwhere(extremas_array == point_to_remove)[0][0]

                if not set(ls_copy).issubset(set(counts_to_remove)):
                    if point_to_remove_index == idx:
                        counts_to_remove.extend((count_list[idx - 1], n))
                    else:
                        if idx == len(count_list) - 1:
                            counts_to_remove.append(n)
                        else:
                            counts_to_remove.extend((n, count_list[idx + 1]))
                    # print(counts_to_remove)
                ls = ls[1:]
                count_list[idx] = 0
                # print('count list after: ' + str(count_list))
        # print('\n')

        for n in counts_to_remove:
            if n in count_list:
                count_list.remove(n)
        points_to_delete = list(dict.fromkeys(points_to_delete))
        # print('points to delete: ' + str(points_to_delete))
        # print('size: ' + str(len(points_to_delete)))
        index_list = [np.argwhere(extremas_array == n)[0][0] for n in points_to_delete]
        extremas_array = np.delete(extremas_array, index_list)
        # print('New extrema array: ' + str(extremas_array))
        # print('size: ' + str(extremas_array.size))

    return extremas_array


# filter extrema points by average angle change
def filter_extremas(angles_array, extremas_array, maxima=True, recursion1=True, recursion2=True):
    if recursion1:
        if extremas_array.size < 2:
            return extremas_array
        angle_diffs = np.absolute(np.diff(extremas_array))
        #print("angle_diffs", angle_diffs)
        # print('extremas: ' + str(extremas_array))
        average_change = float(sum(angle_diffs) / len(angle_diffs))
        # print('averange angle change:', average_change)
        # not sure if this is a good threshold
        # 10 is low, 15 ?
        angle_threshold = average_change + 20
        # print('angle change threshold: ' + str(angle_threshold))
        to_be_removed = []

        if average_change > 12:
            # print('\n')
            # print('Testing to filter by average angle change....')
            size = extremas_array.size
            for n in range(size):
                if n + 1 == size:
                    break
                point1 = extremas_array[n]
                point2 = extremas_array[n + 1]
                max_angle = max(point1, point2)
                min_angle = min(point1, point2)

                if maxima:
                    point_to_remove = min_angle
                else:
                    point_to_remove = max_angle

                if float(max_angle - min_angle) > angle_threshold:
                    to_be_removed.append(np.argwhere(extremas_array == point_to_remove)[0][0])

        removed_by_angle_change = [extremas_array[n] for n in to_be_removed]
        removed_by_angle_change = list(dict.fromkeys(removed_by_angle_change))
        if len(removed_by_angle_change) > 0:
            # print('Testing Finished')
            # print('Filtering by average angle change...')
            # print('extrema array size: ' + str(size))
            # print('angles removed: ' + str(removed_by_angle_change))
            extremas_array = np.delete(extremas_array, to_be_removed)
            # print('\nNew extrema array: ' + str(extremas_array))
            # print('new size: ' + str(extremas_array.size))

            angle_diffs2 = np.absolute(np.diff(extremas_array))
            average_change2 = float(sum(angle_diffs2) / len(angle_diffs2))
            if average_change2 > 12:
                # recurse to check again
                extremas_array = filter_extremas(angles_array, extremas_array, maxima, recursion2=False)
                # else:
            # print('Filter by angle change N/A')

    if recursion2:
        # print('\nTesting to filter by number of angles inbetween....')
        count_list = count_angles_between_extremas(angles_array, extremas_array)
        # print('count list: ' + str(count_list))
        avr_count = int(sum(count_list) / len(count_list))
        threshold = int(avr_count / 2) + len(count_list)
        # print('threshold: ' + str(threshold))
        # print('average count: ' + str(avr_count))
        ls = [n for n in count_list if n < threshold]

        if len(ls) > 0:
            # print(ls)
            # old threshold - 25
            if avr_count - min(ls) > 15:
                # print('Testing Finished')
                # print('Filtering by number of angles inbetween:')
                # print('count_list: ' + str(count_list))
                # print('threshold: ' + str(threshold))
                extremas_array = filter_extrema_by_angles_number_inbetween(extremas_array, count_list, ls, maxima)

                # recursion
                # test if array needs to be filtered again
                count_list2 = count_angles_between_extremas(angles_array, extremas_array)
                avr_count2 = int(sum(count_list2) / len(count_list2))
                threshold2 = int(avr_count2 / 2) + len(count_list2)
                ls2 = [n for n in count_list2 if n < threshold2]
                if len(ls2) > 0:
                    if avr_count2 - min(ls2) > 25:
                        # print('\nStarting another stage of filtering...\n')
                        extremas_array = filter_extremas(angles_array, extremas_array, maxima, recursion1=False)
        # else:
        # print('Filtering by number of angles N/A')

        # if recursion2:
        # print('Array filtering: Done')

    return extremas_array


# local minima points are minimum angles in each rep, no need to calc again
def analyse_each_rep(exercise, mode, extremas1, uf_angles1, ut_angles1, tk_angles1=None, extremas2=None,
                     uf_angles2=None, ut_angles2=None):
    if len(extremas1) == 0:
        return None

    min_upper_arm_forearm = []
    max_upper_arm_forearm = []
    min_upper_arm_trunk = []
    max_upper_arm_trunk = []
    min_trunk_knee = []
    max_trunk_knee = []
    evaluation = {}

    # if one side exercise analysis
    if extremas2 is None:
        uf_points, ut_points, tk_points = [], [], []
        angles_each_rep = []
        rep_count = 0
        max_counter = 0
        count_angles_end = 0
        all_reps = {}
        extremas_copy = extremas1
        num = extremas1.size
        uf_count, ut_count, tk_count = 0, 0, 0
        saved_extremas = []
        uf_df, ut_df, tk_df = [], [], []

        if exercise == 'bicep_curl' or exercise == 'triceps_pushdown':
            count_angles_start = count_angles_between_two_points(extremas1[:1][0], uf_angles1[:1][0], uf_angles1)
            if count_angles_start < 20:
                extremas1 = np.delete(extremas1, 0)

            for (uf_p, ut_p, tk_p) in zip(uf_angles1, ut_angles1, tk_angles1):
                if uf_p not in extremas1:
                    uf_points.append(uf_p)
                    ut_points.append(ut_p)
                    tk_points.append(tk_p)
                else:
                    if uf_p == saved_extremas[-1:]:
                        continue

                    if max_counter == num:
                        uf_points.append(uf_p)
                        ut_points.append(ut_p)
                        tk_points.append(tk_p)
                    else:
                        saved_extremas.append(uf_p)
                        index = np.where(extremas1 == uf_p)
                        extremas1 = np.delete(extremas1, index[0][0])
                        uf_points.append(uf_p)

                        rep_count += 1
                        max_counter += 1
                        min_upper_arm_forearm.append(min(uf_points))
                        max_upper_arm_forearm.append(max(uf_points))
                        min_upper_arm_trunk.append(min(ut_points))
                        max_upper_arm_trunk.append(max(ut_points))
                        min_trunk_knee.append(min(tk_points))
                        max_trunk_knee.append(max(tk_points))

                        all_reps[rep_count] = [
                            "Starting position upper arm forearm angle: " + str(uf_points[:1][0]),
                            "Starting position upper arm trunk angle: " + str(ut_points[:1][0]),
                            "Starting position trunk knee angle: " + str(tk_points[:1][0]) + '\n',
                            "Maximum angle between upper arm and forearm: " + str(max(uf_points)),
                            "Maximum angle between upper arm and trunk: " + str(max(ut_points)),
                            "Maximum angle between trunk and knee: " + str(max(tk_points)),
                            "Minimum angle between trunk and knee: " + str(min(tk_points)),
                            "Minimum angle between upper arm and trunk: " + str(min(ut_points)),
                            "Minimum angle between upper arm and forearm: " + str(min(uf_points)) + '\n',
                            "Finishing position upper arm forearm angle: " + str(uf_points[-1:][0]),
                            "Finishing  position upper arm trunk angle: " + str(ut_points[-1:][0]),
                            "Finishing  position trunk knee angle: " + str(tk_points[-1:][0])
                        ]

                        evaluation[rep_count] = {
                            "start upper arm forearm": uf_points[:1][0],
                            "start upper arm trunk": ut_points[:1][0],
                            "start trunk knee": tk_points[:1][0],
                            'max upper arm forearm': max(uf_points),
                            'max upper arm trunk': max(ut_points),
                            'max trunk knee': max(tk_points),
                            'min trunk knee': min(tk_points),
                            'min upper arm trunk': min(ut_points),
                            'min upper arm forearm': min(uf_points),
                            'finish upper arm forearm': uf_points[-1:][0],
                            'finish upper arm trunk': ut_points[-1:][0],
                            'finish trunk knee': tk_points[-1:][0]
                        }

                        uf_df.append(np.array(uf_points))
                        ut_df.append(np.array(ut_points))
                        tk_df.append(np.array(tk_points))
                        # angles_each_rep.extend((np.array(uf_points), np.array(ut_points), np.array(tk_points)))
                        # erase lists
                        uf_points, ut_points, tk_points = [], [], []

            count_angles_end = count_angles_between_two_points(extremas_copy[-1:][0], uf_angles1[-1:][0], uf_angles1)

        elif exercise == 'front_raise':
            count_angles_start = count_angles_between_two_points(extremas_copy[:1][0], ut_angles1[:1][0], ut_angles1)
            if count_angles_start < 20:
                extremas1 = np.delete(extremas1, 0)

            for (uf_p, ut_p, tk_p) in zip(uf_angles1, ut_angles1, tk_angles1):
                if ut_p not in extremas1:
                    uf_points.append(uf_p)
                    ut_points.append(ut_p)
                    tk_points.append(tk_p)
                else:
                    if max_counter == num:
                        uf_points.append(uf_p)
                        ut_points.append(ut_p)
                        tk_points.append(tk_p)
                    else:
                        index = np.where(extremas1 == ut_p)
                        extremas1 = np.delete(extremas1, index[0][0])
                        ut_points.append(ut_p)

                        rep_count += 1
                        max_counter += 1

                        min_upper_arm_forearm.append(min(uf_points))
                        max_upper_arm_forearm.append(max(uf_points))
                        min_upper_arm_trunk.append(min(ut_points))
                        max_upper_arm_trunk.append(max(ut_points))
                        min_trunk_knee.append(min(tk_points))
                        max_trunk_knee.append(max(tk_points))

                        all_reps[rep_count] = [
                            "Starting position upper arm forearm angle: " + str(uf_points[:1][0]),
                            "Starting position upper arm trunk angle: " + str(ut_points[:1][0]),
                            "Starting position trunk knee angle: " + str(tk_points[:1][0]) + '\n',
                            "Maximum angle between upper arm and forearm: " + str(max(uf_points)),
                            "Maximum angle between upper arm and trunk: " + str(max(ut_points)),
                            "Maximum angle between trunk and knee: " + str(max(tk_points)),
                            "Minimum angle between upper arm trunk: " + str(min(ut_points)),
                            "Minimum angle between trunk and knee: " + str(min(tk_points)),
                            "Minimum angle between upper arm and forearm: " + str(min(uf_points)) + '\n',
                            "Finishing position upper arm forearm angle: " + str(uf_points[-1:][0]),
                            "Finishing  position upper arm trunk angle: " + str(ut_points[-1:][0]),
                            "Finishing  position trunk knee angle: " + str(tk_points[-1:][0])
                        ]

                        evaluation[rep_count] = {
                            "start upper arm forearm": uf_points[:1][0],
                            "start upper arm trunk": ut_points[:1][0],
                            "start trunk knee": tk_points[:1][0],
                            'max upper arm forearm': max(uf_points),
                            'max upper arm trunk': max(ut_points),
                            'max trunk knee': max(tk_points),
                            'min trunk knee': min(tk_points),
                            'min upper arm trunk': min(ut_points),
                            'min upper arm forearm': min(uf_points),
                            'finish upper arm forearm': uf_points[-1:][0],
                            'finish upper arm trunk': ut_points[-1:][0],
                            'finish trunk knee': tk_points[-1:][0]
                        }

                        uf_df.append(np.array(uf_points))
                        ut_df.append(np.array(ut_points))
                        tk_df.append(np.array(tk_points))
                        # angles_each_rep.extend((np.array(uf_points), np.array(ut_points), np.array(tk_points)))
                        # erase lists
                        uf_points, ut_points, tk_points = [], [], []

            count_angles_end = count_angles_between_two_points(extremas_copy[-1:][0], ut_angles1[-1:][0], ut_angles1)

        if count_angles_end > 20:
            # Last rep analysis
            rep_count += 1

            min_upper_arm_forearm.append(min(uf_points))
            max_upper_arm_forearm.append(max(uf_points))
            min_upper_arm_trunk.append(min(ut_points))
            max_upper_arm_trunk.append(max(ut_points))
            min_trunk_knee.append(min(tk_points))
            max_trunk_knee.append(max(tk_points))

            all_reps[rep_count] = [
                "Starting position upper arm forearm angle: " + str(uf_points[:1][0]),
                "Starting position upper arm trunk angle: " + str(ut_points[:1][0]),
                "Starting position trunk knee angle: " + str(tk_points[:1][0]) + '\n',
                "Maximum angle between upper arm and forearm: " + str(max(uf_points)),
                "Maximum angle between upper arm and trunk: " + str(max(ut_points)),
                "Maximum angle between trunk and knee: " + str(max(tk_points)),
                "Minimum angle between upper arm trunk: " + str(min(ut_points)),
                "Minimum angle between trunk and knee: " + str(min(tk_points)),
                "Minimum angle between upper arm and forearm: " + str(min(uf_points)) + '\n',
                "Finishing position upper arm forearm angle: " + str(uf_points[-1:][0]),
                "Finishing  position upper arm trunk angle: " + str(ut_points[-1:][0]),
                "Finishing  position trunk knee angle: " + str(tk_points[-1:][0])
            ]

            evaluation[rep_count] = {
                "start upper arm forearm": uf_points[:1][0],
                "start upper arm trunk": ut_points[:1][0],
                "start trunk knee": tk_points[:1][0],
                'max upper arm forearm': max(uf_points),
                'max upper arm trunk': max(ut_points),
                'max trunk knee': max(tk_points),
                'min trunk knee': min(tk_points),
                'min upper arm trunk': min(ut_points),
                'min upper arm forearm': min(uf_points),
                'finish upper arm forearm': uf_points[-1:][0],
                'finish upper arm trunk': ut_points[-1:][0],
                'finish trunk knee': tk_points[-1:][0]
            }

            uf_df.append(np.array(uf_points))
            ut_df.append(np.array(ut_points))
            tk_df.append(np.array(tk_points))
            # angles_each_rep.extend((np.array(uf_points), np.array(ut_points), np.array(tk_points)))

        if mode == 'dataset':
            # angles_each_rep
            return uf_df, ut_df, tk_df
        elif mode == 'analysis':
            print('Number of reps performed: ' + str(rep_count))
            for k, v in all_reps.items():
                print('\n' + 'Repetition: ' + str(k) + '\n')
                for s in v:
                    print(s)
        elif mode == 'evaluation':
            return evaluation

        elif mode == 'thresholds':
            return min_upper_arm_forearm, max_upper_arm_forearm, min_upper_arm_trunk, max_upper_arm_trunk, min_trunk_knee, max_trunk_knee

        elif mode == 'repetitions':
            return rep_count
        else:
            print('Error: Wrong function mode. Select one of the following: dataset, analysis, evaluation, thresholds,'
                  ' repetitions.')
            return None

    elif extremas2 is not None:
        left_uf_points, right_uf_points, left_ut_points, right_ut_points = [], [], [], []
        left_side_angles, right_side_angles = [], []
        each_rep_angles = []
        left_rep_count, right_rep_count = 0, 0
        left_max_counter, right_max_counter = 0, 0
        count_left_end, count_right_end = 0, 0

        both_uf_angles, both_ut_angles = [], []

        min_upper_arm_forearm = []
        max_upper_arm_forearm = []
        max_upper_arm_trunk = []
        min_upper_arm_trunk = []

        evaluation_left = {}
        evaluation_right = {}

        left_reps, right_reps = {}, {}
        left_extremas = extremas1
        right_extremas = extremas2
        num1 = extremas1.size
        num2 = extremas2.size
        l_uf_count, l_ut_count = 0, 0
        r_uf_count, r_ut_count = 0, 0

        if abs(num1 - num2) > 1:
            print("Left and Right extrema points not equal")
        if exercise == 'shoulder_press':
            count_left_start = count_angles_between_two_points(left_extremas[:1], ut_angles1[:1], ut_angles1)
            count_right_start = count_angles_between_two_points(right_extremas[:1], ut_angles2[:1], ut_angles2)

            if count_left_start < 20:
                extremas1 = np.delete(extremas1, 0)

            if count_right_start < 20:
                extremas2 = np.delete(extremas2, 0)

            for (left_uf_p, right_uf_p, left_ut_p, right_ut_p) in zip(uf_angles1, uf_angles2, ut_angles1, ut_angles2):
                if left_ut_p not in extremas1:
                    left_uf_points.append(left_uf_p)
                    left_ut_points.append(left_ut_p)
                else:
                    if left_max_counter == num1:
                        left_uf_points.append(left_uf_p)
                        left_ut_points.append(left_ut_p)
                    else:
                        # fix for duplicates
                        index = np.where(extremas1 == left_ut_p)
                        extremas1 = np.delete(extremas1, index[0][0])
                        left_uf_points.append(left_uf_p)
                        left_rep_count += 1
                        left_max_counter += 1

                        min_upper_arm_forearm.append(min(left_uf_points))
                        max_upper_arm_forearm.append(max(left_uf_points))
                        max_upper_arm_trunk.append(max(left_ut_points))
                        min_upper_arm_trunk.append(min(left_ut_points))

                        left_reps[left_rep_count] = [
                            'Starting position left upper arm - left forearm angle: ' + str(left_uf_points[:1][0]),
                            'Starting position left upper arm - trunk angle: ' + str(left_ut_points[:1][0]) + '\n',
                            'Left upper arm - left forearm -> Minimum Angle: ' + str(min(left_uf_points)),
                            'Left upper arm - left forearm -> Maximum Angle: ' + str(max(left_uf_points)),
                            'Left upper arm - trunk -> Maximum Angle: ' + str(max(left_ut_points)),
                            'Left upper arm - trunk -> Minimum Angle: ' + str(min(left_ut_points)) + '\n',
                            'Finishing position left upper arm - left forearm angle: ' + str(left_uf_points[-1:][0]),
                            'Finishing position left upper arm - trunk angle: ' + str(left_ut_points[-1:][0])
                        ]

                        evaluation_left[left_rep_count] = {
                            'start left upper arm forearm': left_uf_points[:1][0],
                            'start left upper arm trunk': left_ut_points[:1][0],
                            'min left upper arm forearm': min(left_uf_points),
                            'min left upper arm trunk': min(left_ut_points),
                            'max left upper arm trunk': max(left_ut_points),
                            'max left upper arm forearm': max(left_uf_points),
                            'finish left upper arm forearm': left_uf_points[-1:][0],
                            'finish left upper arm trunk': left_ut_points[-1:][0]
                        }

                        both_uf_angles.append(np.array(left_uf_points))
                        both_ut_angles.append(np.array(left_ut_points))

                        # left_side_angles.extend((np.array(left_uf_points), np.array(left_ut_points)))
                        l_uf_count += 1
                        l_ut_count += 1

                        # erase lists
                        left_uf_points, left_ut_points = [], []

                if right_ut_p not in extremas2:
                    right_uf_points.append(right_uf_p)
                    right_ut_points.append(right_ut_p)
                else:
                    if right_max_counter == num2:
                        right_uf_points.append(right_uf_p)
                        right_ut_points.append(right_ut_p)
                    else:
                        index = np.where(extremas2 == right_ut_p)
                        extremas2 = np.delete(extremas2, index[0][0])
                        right_uf_points.append(right_uf_p)
                        right_max_counter += 1
                        right_rep_count += 1

                        min_upper_arm_forearm.append(min(right_uf_points))
                        max_upper_arm_forearm.append(max(right_uf_points))
                        max_upper_arm_trunk.append(max(right_ut_points))
                        min_upper_arm_trunk.append(min(right_ut_points))

                        right_reps[right_rep_count] = [
                            'Starting position right upper arm - right forearm angle: ' + str(right_uf_points[:1][0]),
                            'Starting position right upper arm - trunk angle: ' + str(right_ut_points[:1][0]) + '\n',
                            'Right upper arm - right forearm -> Minimum Angle: ' + str(min(right_uf_points)),
                            'Right upper arm - right forearm -> Maximum Angle: ' + str(max(right_uf_points)),
                            'Right upper arm - trunk -> Maximum Angle: ' + str(max(right_ut_points)),
                            'Right upper arm - trunk -> Minimum Angle: ' + str(min(right_ut_points)) + '\n',
                            'Finishing position right upper arm - right forearm angle: ' + str(right_uf_points[-1:][0]),
                            'Finishing position right upper arm - trunk angle: ' + str(right_ut_points[-1:][0])

                        ]

                        evaluation_right[right_rep_count] = {
                            'start right upper arm forearm': right_uf_points[:1][0],
                            'start right upper arm trunk': right_ut_points[:1][0],
                            'min right upper arm forearm': min(right_uf_points),
                            'min right upper arm trunk': min(right_ut_points),
                            'max right upper arm trunk': max(right_ut_points),
                            'max right upper arm forearm': max(right_uf_points),
                            'finish right upper arm forearm': right_uf_points[-1:][0],
                            'finish right upper arm trunk': right_ut_points[-1:][0]
                        }

                        both_uf_angles.append(np.array(right_uf_points))
                        both_ut_angles.append(np.array(right_ut_points))
                        # right_side_angles.extend((np.array(right_uf_points), np.array(right_ut_points)))
                        r_uf_count += 1
                        r_ut_count += 1
                        # erase lists
                        right_uf_points, right_ut_points = [], []

            count_left_end = count_angles_between_two_points(left_extremas[-1:], ut_angles1[-1:], ut_angles1)
            count_right_end = count_angles_between_two_points(right_extremas[-1:], ut_angles2[-1:], ut_angles2)
            # print('angles between left last extrema and angle: ' + str(count_left))
            # print('angles between right last extrema and angle: ' + str(count_right))

        if count_left_end > 20:
            # Last rep analysis
            left_rep_count += 1

            min_upper_arm_forearm.append(min(left_uf_points))
            max_upper_arm_forearm.append(max(left_uf_points))
            max_upper_arm_trunk.append(max(left_ut_points))
            min_upper_arm_trunk.append(min(left_ut_points))

            left_reps[left_rep_count] = [
                'Starting position left upper arm - left forearm angle: ' + str(left_uf_points[:1][0]),
                'Starting position left upper arm - trunk angle: ' + str(left_ut_points[:1][0]) + '\n',
                'Left upper arm - left forearm -> Minimum Angle: ' + str(min(left_uf_points)),
                'Left upper arm - left forearm -> Maximum Angle: ' + str(max(left_uf_points)),
                'Left upper arm - trunk -> Maximum Angle: ' + str(max(left_ut_points)),
                'Left upper arm - trunk -> Minimum Angle: ' + str(min(left_ut_points)) + '\n',
                'Finishing position left upper arm - left forearm angle: ' + str(left_uf_points[-1:][0]),
                'Finishing position left upper arm - trunk angle: ' + str(left_ut_points[-1:][0])
            ]

            evaluation_left[left_rep_count] = {
                'start left upper arm forearm': left_uf_points[:1][0],
                'start left upper arm trunk': left_ut_points[:1][0],
                'min left upper arm forearm': min(left_uf_points),
                'min left upper arm trunk': min(left_ut_points),
                'max left upper arm trunk': max(left_ut_points),
                'max left upper arm forearm': max(left_uf_points),
                'finish left upper arm forearm': left_uf_points[-1:][0],
                'finish left upper arm trunk': left_ut_points[-1:][0]
            }

            both_uf_angles.append(np.array(left_uf_points))
            both_ut_angles.append(np.array(left_ut_points))
            # left_side_angles.extend((np.array(left_uf_points), np.array(left_ut_points)))
            l_uf_count += 1
            l_ut_count += 1

        if count_right_end > 20:
            right_rep_count += 1

            min_upper_arm_forearm.append(min(right_uf_points))
            max_upper_arm_forearm.append(max(right_uf_points))
            max_upper_arm_trunk.append(max(right_ut_points))
            min_upper_arm_trunk.append(min(right_ut_points))

            right_reps[right_rep_count] = [
                'Starting position right upper arm - right forearm angle: ' + str(right_uf_points[:1][0]),
                'Starting position right upper arm - trunk angle: ' + str(right_ut_points[:1][0]) + '\n',
                'Right upper arm - right forearm -> Minimum Angle: ' + str(min(right_uf_points)),
                'Right upper arm - right forearm -> Maximum Angle: ' + str(max(right_uf_points)),
                'Right upper arm - trunk -> Maximum Angle: ' + str(max(right_ut_points)),
                'Right upper arm - trunk -> Minimum Angle: ' + str(min(right_ut_points)) + '\n',
                'Finishing position right upper arm - right forearm angle: ' + str(right_uf_points[-1:][0]),
                'Finishing position right upper arm - trunk angle: ' + str(right_ut_points[-1:][0])

            ]

            evaluation_right[right_rep_count] = {
                'start right upper arm forearm': right_uf_points[:1][0],
                'start right upper arm trunk': right_ut_points[:1][0],
                'min right upper arm forearm': min(right_uf_points),
                'min right upper arm trunk': min(right_ut_points),
                'max right upper arm trunk': max(right_ut_points),
                'max right upper arm forearm': max(right_uf_points),
                'finish right upper arm forearm': right_uf_points[-1:][0],
                'finish right upper arm trunk': right_ut_points[-1:][0]
            }

            both_uf_angles.append(np.array(right_uf_points))
            both_ut_angles.append(np.array(right_ut_points))
            # right_side_angles.extend((np.array(right_uf_points), np.array(right_ut_points)))
            r_uf_count += 1
            r_ut_count += 1

        all_reps = {}
        evaluation_both_arms = {}
        # print('left rep count: ' + str(left_rep_count))
        # print('right rep count: ' + str(right_rep_count))
        # Combine dicts for all reps
        if left_rep_count == right_rep_count and mode != 'repetitions':
            # combine two dictionaries
            ds = [left_reps, right_reps]
            for key in left_reps.keys():
                all_reps[key] = tuple(d[key] for d in ds)

            eval_dicts = [evaluation_left, evaluation_right]
            for key in evaluation_left.keys():
                evaluation_both_arms[key] = tuple(d[key] for d in eval_dicts)
        else:
            print("Error: Rep counts for left and right arms are not equal")

        if mode == 'dataset':
            if len(left_side_angles) == len(right_side_angles):
                # left_side_angles + right_side_angles, (l_uf_count + l_ut_count, r_uf_count + r_ut_count)
                return both_uf_angles, both_ut_angles
            else:
                print('Left and Right side anlges are not equal!')

        elif mode == 'analysis':
            print('Number of reps performed: ' + str(left_rep_count))
            for k, v in all_reps.items():
                print('\n' + 'Repetition: ' + str(k) + '\n')
                for s in v:
                    for m in s:
                        print(str(m))
        elif mode == 'evaluation':
            return evaluation_both_arms

        elif mode == 'thresholds':
            return min_upper_arm_forearm, max_upper_arm_forearm, min_upper_arm_trunk, max_upper_arm_trunk

        elif mode == 'repetitions':
            if left_rep_count == right_rep_count:
                return left_rep_count
            else:
                return 0

        else:
            print('Error: Wrong function mode. Select one of the following: dataset, analysis, evaluation, thresholds,'
                  ' repetitions.')


def get_evaluation_decision(feedback, rep_count, display=True):
    final_feedback = {'Good': [], 'Bad': []}
    if not feedback:
        return 'No Feedback: Error happened during analysis.'

    if display:
        print('\nAnalysing each repetition below: ')

        for fs in feedback:
            print('\n')
            print('-' * 100)
            print('Repetition ' + str(fs[1]))
            print('-' * 100)
            for fb in fs[0]:
                print(fb)

            if all('Good' in fbs for fbs in fs[0]):
                rep_feedback = '\nRepetition was performed with a good form.'
                print(rep_feedback)
                final_feedback['Good'].append(fs[1])
            else:
                print('\nRepetition form could be improved.')
                final_feedback['Bad'].append(fs[1])

        print('\n')
        print('-' * 100)
        if len(final_feedback['Good']) == rep_count:
            decision = 'Decision: Correct Form! Exercise was performed with a correct technique.'

        else:
            if len(final_feedback['Bad']) <= rep_count - int(rep_count * (8 / 10)):  # 80% of the total rep count
                decision = 'Decision: Overall the exercise was performed with a good form. However, your technique during ' \
                           'the following reps: ' + str(
                    final_feedback['Bad']) + ' could be improved. Look at the feedback ' \
                                             'for the following reps to see what you did wrong.'

            else:
                decision = 'Decision: Incorrect Form! Your technique can be improved. The following reps have been performed with a bad form: ' + str(
                    final_feedback['Bad']) + '. Take a look at the feedback ' \
                                             'above to see what you did wrong.'

        print(decision)
        print('-' * 100)

    else:
        for fs in feedback:
            if all('Good' in fbs for fbs in fs[0]):
                final_feedback['Good'].append(fs[1])
            else:
                final_feedback['Bad'].append(fs[1])

        if len(final_feedback['Good']) == rep_count:
            return 0
        else:
            return 1


# Fill na values
def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=np.float)
    out[mask] = np.concatenate(data)
    return out
