import numpy as np
import math
import itertools


# Detecting sides (left or right) on videos
# can use this method for other exercises
def detect_side(frame_poses):
    rside_joints = [1 for pose in frame_poses if pose.joint_keypoints['RSHOULDER'][2] != 0 and
                    pose.joint_keypoints['RELBOW'][2] != 0 and
                    pose.joint_keypoints['RWRIST'][2] != 0 and
                    pose.joint_keypoints['RHIP'][2] != 0 and
                    pose.joint_keypoints['RKNEE'][2] != 0]

    lside_joints = [1 for pose in frame_poses if pose.joint_keypoints['LSHOULDER'][2] != 0 and
                    pose.joint_keypoints['LELBOW'][2] != 0 and
                    pose.joint_keypoints['LWRIST'][2] != 0 and
                    pose.joint_keypoints['LHIP'][2] != 0 and
                    pose.joint_keypoints['LKNEE'][2] != 0]

    print('Right side: ' + str(sum(rside_joints)))
    print('Left  side: ' + str(sum(lside_joints)))

    # think about the case when they are equal
    if sum(rside_joints) > sum(lside_joints):
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
    return extrema_points, indexes


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
def find_frames_between_extremas(extrema1, extrema2, angles_array):
    index1 = np.argwhere(angles_array == extrema1)[0][0]
    index2 = np.argwhere(angles_array == extrema2)[0][0]
    nums = np.arange(index1 + 1, index2)
    print(index1, index2)
    pm = np.take(angles_array, nums)
    count = pm.size

    return count


# filter extrema points
def filter_extremas(angles_array, extremas_array, maxima=False):
    if maxima = False:
        count1 = 0

        # do not count points before first extrema point
        indexes = np.argwhere(angles_array == extremas_array[0])
        print('first extrema index: ' + str(indexes[0][0]))
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
        print('Extrema array size: ' + str(extremas_array.size))
        print('Count list:' + str(count_list))
        print('length count list: ' + str(len(count_list)))
        count_diffs = np.absolute(np.diff(np.array(count_list)))
        threshold = max(count_list) - min(count_diffs) - len(count_list)
        print('threshold is: ' + str(threshold))
        ls = [n for n in count_list if n < threshold]
        print("ls size: " + str(len(ls)))
        print("ls list: " + str(ls))
        print(extremas_array)
        if len(ls) > 0:
            indxs_to_delete = [count_list.index(n) for n in ls]
            print('indexes to delete: ' + str(indxs_to_delete))
            print('size: ' + str(len(indxs_to_delete)))
            size1 = extremas_array.size
            extremas_array_copy = extremas_array

            for idx in indxs_to_delete:
                if idx + 1 < size1:
                    point1 = extremas_array_copy[idx]
                    point2 = extremas_array_copy[idx + 1]
                    start_point = min(point1, point2)
                    remove_point = max(point1, point2)
                    b = np.argwhere(extremas_array_copy == remove_point)[0][0]
                    print('remove point '+str(b))
                    extremas_array = np.delete(extremas_array, b)
                    indx = np.argwhere(extremas_array == start_point)[0][0]
                    print('index ' + str(indx))
                    count = find_frames_between_extremas(start_point, extremas_array[indx+1], angles_array)
                    count_list.insert(idx, count)

            length = extremas_array.size
            print('length new array: ' + str(length))
            print("new array: " + str(extremas_array))
''

    return extremas_array


# Could modify this function to use across other exercise classes
# local minima points are minimum angles in each rep, no need to calc again
def analyse_each_rep(string, extremas1, uf_angles1a, ut_angles2a, tk_angles3a=None, extremas2=None, uf_angles1b=None,
                     ut_angles2b=None):
    if extremas1.size == 0:
        return None
    if extremas2 is not None:
        if extremas2.size == 0:
            return None

    if tk_angles3a is not None:
        list_maxes, uf_points, ut_points, tk_points = [], [], [], []
        angles_each_rep = []
        rep_count = 0
        max_counter = 0
        all_reps = {}

        print("angles1 size: " + str(uf_angles1a.size))
        print("angles2 size: " + str(ut_angles2a.size))
        print("angles3 size: " + str(tk_angles3a.size))
        count = 0
        ncount = 0
        num = extremas1.size
        for (uf_p, ut_p, tk_p) in zip(uf_angles1a, ut_angles2a, tk_angles3a):
            ncount += 1
            # print('Count ' + str(ncount) + 'out of ' + str(ut_angles2a.size))
            if uf_p not in extremas1:
                uf_points.append(uf_p)
                ut_points.append(ut_p)
                tk_points.append(tk_p)
            else:
                if max_counter == num:
                    uf_points.append(uf_p)
                    ut_points.append(ut_p)
                    tk_points.append(tk_p)
                else:
                    count += 1
                    print('Got ' + str(count) + ' out of ' + str(num))
                    index = np.where(extremas1 == uf_p)
                    extremas1 = np.delete(extremas1, index[0][0])
                    uf_points.append(uf_p)
                    rep_count += 1
                    max_counter += 1
                    all_reps[rep_count] = [
                        "Minimum angle between upper arm and forearm: " + str(min(uf_points)),
                        "Maximum angle between upper arm and trunk: " + str(max(ut_points)),
                        "Minimum angle between trunk and knee: " + str(min(tk_points))]

                    # then do if statements to check if angles above/below threshold
                    angles_each_rep.extend((np.array(uf_points), np.array(ut_points), np.array(tk_points)))
                    # erase lists
                    uf_points, ut_points, tk_points = [], [], []

        print('Loop exited\n\n')
        print('Extrema size' + str(extremas1.size))
        print('Reps:' + str(rep_count))

        # Last rep analysis
        rep_count += 1
        all_reps[rep_count] = [
            "Minimum angle between upper arm and forearm: " + str(min(uf_points)),
            "Maximum angle between upper arm and trunk: " + str(max(ut_points)),
            "Minimum angle between trunk and knee: " + str(min(tk_points))]

        angles_each_rep.extend((np.array(uf_points), np.array(ut_points), np.array(tk_points)))
        if string == 'dataset':
            return angles_each_rep
        elif string == 'analysis':
            for k, v in all_reps.items():
                print('\n' + 'Repetition: ' + str(k) + '\n')
                for s in v:
                    print(s)

    else:
        list_maxes_left, list_maxes_right, left_uf_points, right_uf_points, left_ut_points, right_ut_points = [], [], [], [], [], []
        angles_each_rep_left, angles_each_rep_right = [], []
        left_rep_count, right_rep_count = 0, 0
        left_max_counter, right_max_counter = 0, 0

        left_reps, right_reps = {}, {}
        left_extremas = extremas1
        right_extremas = extremas2
        num1 = extremas1.size
        num2 = extremas2.size

        if num1 != num2:
            print("Left and Right maxima points not equal")

        for (left_uf_p, right_uf_p, left_ut_p, right_ut_p) in zip(uf_angles1a, uf_angles1b, ut_angles2a, ut_angles2b):
            if left_uf_p not in left_extremas:
                left_uf_points.append(left_uf_p)
                left_ut_points.append(left_ut_p)
            else:
                if left_max_counter == num1:
                    left_uf_points.append(left_uf_p)
                    left_ut_points.append(left_ut_p)
                else:
                    # fix for duplicates
                    index = np.where(left_extremas == left_uf_p)
                    left_extremas = np.delete(left_extremas, index[0][0])
                    left_uf_points.append(left_uf_p)
                    left_rep_count += 1
                    left_max_counter += 1
                    left_reps[left_rep_count] = [
                        'Left upper arm - left forearm -> Minimum Angle:' + str(min(left_uf_points)),
                        'Left upper arm - left forearm -> Maximum Angle:' + str(max(left_uf_points)),
                        'Left upper arm - trunk -> Maximum Angle: ' + str(max(left_ut_points)),
                        'Left upper arm - trunk -> Minimum Angle: ' + str(min(left_ut_points)) + '\n']

                    # then do if statements to check if angles above/below threshold
                    angles_each_rep_left.extend((np.array(left_uf_points), np.array(left_ut_points)))
                    # erase lists
                    left_uf_points, left_ut_points = [], []

            if right_uf_p not in right_extremas:
                right_uf_points.append(right_uf_p)
                right_ut_points.append(right_ut_p)
            else:
                if right_max_counter == num2:
                    right_uf_points.append(right_uf_p)
                    right_ut_points.append(right_ut_p)
                else:
                    index = np.where(right_extremas == right_uf_p)
                    right_extremas = np.delete(right_extremas, index[0][0])
                    right_uf_points.append(right_uf_p)
                    right_max_counter += 1
                    right_rep_count += 1
                    right_reps[right_rep_count] = [
                        'Right upper arm - left forearm -> Minimum Angle:' + str(min(right_uf_points)),
                        'Right upper arm - left forearm -> Maximum Angle:' + str(max(right_uf_points)),
                        'Right upper arm - trunk -> Maximum Angle: ' + str(max(right_ut_points)),
                        'Right upper arm - trunk -> Minimum Angle: ' + str(min(right_ut_points)) + '\n']

                    # then do if statements to check if angles above/below threshold
                    angles_each_rep_right.extend((np.array(right_uf_points), np.array(right_ut_points)))
                    # erase lists
                    right_uf_points, right_ut_points = [], []

        # Last rep analysis
        left_rep_count += 1
        left_reps[left_rep_count] = [
            'Left upper arm - left forearm -> Minimum Angle:' + str(min(left_uf_points)),
            'Left upper arm - left forearm -> Maximum Angle:' + str(max(left_uf_points)),
            'Left upper arm - trunk -> Maximum Angle: ' + str(max(left_ut_points)),
            'Left upper arm - trunk -> Minimum Angle: ' + str(min(left_ut_points)) + '\n']

        # then do if statements to check if angles above/below threshold
        angles_each_rep_left.extend((np.array(left_uf_points), np.array(left_ut_points)))

        right_rep_count += 1
        right_reps[right_rep_count] = [
            'Right upper arm - left forearm -> Minimum Angle:' + str(min(right_uf_points)),
            'Right upper arm - left forearm -> Maximum Angle:' + str(max(right_uf_points)),
            'Right upper arm - trunk -> Maximum Angle: ' + str(max(right_ut_points)),
            'Right upper arm - trunk -> Minimum Angle: ' + str(min(right_ut_points)) + '\n']

        # then do if statements to check if angles above/below threshold
        angles_each_rep_right.extend((np.array(right_uf_points), np.array(right_ut_points)))

        all_reps = {}
        # Combine dicts for all reps
        if left_rep_count == right_rep_count:
            ds = [left_reps, right_reps]
            for key in left_reps.keys():
                all_reps[key] = tuple(d[key] for d in ds)
        else:
            print(left_rep_count)
            print(right_rep_count)
            print("Rep counts for left and right are not equal")

        if string == 'dataset':
            return angles_each_rep_left, angles_each_rep_right
        elif string == 'analysis':
            for k, v in all_reps.items():
                print('\n' + 'Repetition: ' + str(k) + '\n')
                for s in v:
                    for m in s:
                        print(str(m))
