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
    print('Right side: ' + str(right_sum))
    print('Left  side: ' + str(left_sum))

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
    nums = np.arange(index1 + 1, index2)
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


def filter_extrema_by_angles_number_inbetween(extremas_array, count_list, ls, threshold, maxima=True):
    points_to_delete = []
    counts_to_remove = []
    ls_copy = ls
    size = len(ls)
    print('counts less than threshold: ' + str(ls))
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
def filter_extremas(angles_array, extremas_array, maxima=True, recursion=False):
    if not recursion:
        angle_diffs = np.absolute(np.diff(extremas_array))
        # print('\n')
        print('extremas: ' + str(extremas_array))
        average_change = float(sum(angle_diffs) / len(angle_diffs))
        # not sure if this is a good threshold
        # 10 is low, 15 ?
        angle_threshold = average_change + 20
        print('angle change threshold: ' + str(angle_threshold))

        # if average_change > 15:
        print('\n')
        print('Testing to filter by average angle change....')
        to_be_removed = []
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
            print('Testing Finished')
            print('Filtering by average angle change...')
            print('extrema array size: ' + str(size))
            print('angles removed: ' + str(removed_by_angle_change))
            extremas_array = np.delete(extremas_array, to_be_removed)
            print('\nNew extrema array: ' + str(extremas_array))
            print('new size: ' + str(extremas_array.size))
        else:
            print('Filter by angle change N/A')

    print('\nTesting to filter by number of angles inbetween....')
    count_list = count_angles_between_extremas(angles_array, extremas_array)
    avr_count = int(sum(count_list) / len(count_list))
    threshold = int(avr_count / 2) + len(count_list)
    ls = [n for n in count_list if n < threshold]

    if len(ls) > 0:
        # old threshold - 25
        if avr_count - min(ls) >= 20:
            print('Testing Finished')
            print('Filtering by number of angles inbetween:')
            print('count_list: ' + str(count_list))
            print('threshold: ' + str(threshold))
            extremas_array = filter_extrema_by_angles_number_inbetween(extremas_array, count_list, ls, threshold,
                                                                           maxima)

            # recursion
            # test if array needs to be filtered again
            count_list2 = count_angles_between_extremas(angles_array, extremas_array)
            avr_count2 = int(sum(count_list2) / len(count_list2))
            threshold2 = int(avr_count2 / 2) + len(count_list2)
            ls2 = [n for n in count_list2 if n < threshold2]
            if len(ls2) > 0:
                if avr_count2 - min(ls2) > 25:
                    print('\nStarting another stage of filtering...\n')
                    extremas_array = filter_extremas(angles_array, extremas_array, maxima, recursion=True)

    if not recursion:
        print('Array filtering: Done')

    return extremas_array


# Could modify this function to use across other exercise classes
# local minima points are minimum angles in each rep, no need to calc again
def analyse_each_rep(exercise, string, extremas1, uf_angles1, ut_angles1, tk_angles1=None, extremas2=None, uf_angles2=None, ut_angles2=None, tk_angles2=None):
    if len(extremas1) == 0:
        return None

    # if one side exercise analysis
    if extremas2 is None:
        uf_points, ut_points, tk_points = [], [], []
        angles_each_rep = []
        rep_count = 0
        max_counter = 0
        count_angles = 0
        all_reps = {}
        extremas_copy = extremas1
        num = extremas1.size

        if exercise == 'bicep curl' or exercise == 'triceps pushdown':
            for (uf_p, ut_p, tk_p) in zip(uf_angles1, ut_angles1, tk_angles1):
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
                        index = np.where(extremas1 == uf_p)
                        extremas1 = np.delete(extremas1, index[0][0])
                        uf_points.append(uf_p)

                        rep_count += 1
                        max_counter += 1
                        all_reps[rep_count] = [
                            "Minimum angle between upper arm and forearm: " + str(min(uf_points)),
                            "Maximum angle between upper arm and forearm: " + str(max(uf_points)),
                            "Maximum angle between upper arm and trunk: " + str(max(ut_points)),
                            "Minimum angle between trunk and knee: " + str(min(tk_points))]

                        # then do if statements to check if angles above/below threshold
                        angles_each_rep.extend((np.array(uf_points), np.array(ut_points), np.array(tk_points)))
                        # erase lists
                        uf_points, ut_points, tk_points = [], [], []

            count_angles = count_angles_between_two_points(extremas_copy[-1:][0], uf_angles1[-1:], uf_angles1)

        elif exercise == 'front raise':
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
                        all_reps[rep_count] = [
                            "Minimum angle between upper arm and forearm: " + str(min(uf_points)),
                            "Maximum angle between upper arm and forearm: " + str(max(uf_points)),
                            "Maximum angle between upper arm and trunk: " + str(max(ut_points)),
                            "Minimum angle between trunk and knee: " + str(min(tk_points))]

                        # then do if statements to check if angles above/below threshold
                        angles_each_rep.extend((np.array(uf_points), np.array(ut_points), np.array(tk_points)))
                        # erase lists
                        uf_points, ut_points, tk_points = [], [], []

            count_angles = count_angles_between_two_points(extremas_copy[-1:][0], ut_angles1[-1:], ut_angles1)

        if count_angles > 20:
            # Last rep analysis
            rep_count += 1
            all_reps[rep_count] = [
                "Minimum angle between upper arm and forearm: " + str(min(uf_points)),
                "Maximum angle between upper arm and forearm: " + str(max(uf_points)),
                "Maximum angle between upper arm and trunk: " + str(max(ut_points)),
                "Minimum angle between trunk and knee: " + str(min(tk_points))]

            angles_each_rep.extend((np.array(uf_points), np.array(ut_points), np.array(tk_points)))

        if string == 'dataset':
            return angles_each_rep
        elif string == 'analysis':
            print('Number of reps performed: ' + str(rep_count))
            for k, v in all_reps.items():
                print('\n' + 'Repetition: ' + str(k) + '\n')
                for s in v:
                    print(s)

    elif extremas2 is not None:
        left_uf_points, right_uf_points, left_ut_points, right_ut_points = [], [], [], []
        angles_each_rep_left, angles_each_rep_right = [], []
        left_rep_count, right_rep_count = 0, 0
        left_max_counter, right_max_counter = 0, 0
        count_left, count_right = 0, 0

        left_reps, right_reps = {}, {}
        left_extremas = extremas1
        right_extremas = extremas2
        num1 = extremas1.size
        num2 = extremas2.size

        if abs(num1 - num2) > 1:
            print("Left and Right extrema points not equal")
        if exercise == 'shoulder press':
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
                        left_reps[left_rep_count] = [
                            'Left upper arm - left forearm -> Minimum Angle:' + str(min(left_uf_points)),
                            'Left upper arm - left forearm -> Maximum Angle:' + str(max(left_uf_points)),
                            'Left upper arm - trunk -> Maximum Angle: ' + str(max(left_ut_points)),
                            'Left upper arm - trunk -> Minimum Angle: ' + str(min(left_ut_points)) + '\n']

                        # then do if statements to check if angles above/below threshold
                        angles_each_rep_left.extend((np.array(left_uf_points), np.array(left_ut_points)))
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
                        right_reps[right_rep_count] = [
                            'Right upper arm - left forearm -> Minimum Angle:' + str(min(right_uf_points)),
                            'Right upper arm - left forearm -> Maximum Angle:' + str(max(right_uf_points)),
                            'Right upper arm - trunk -> Maximum Angle: ' + str(max(right_ut_points)),
                            'Right upper arm - trunk -> Minimum Angle: ' + str(min(right_ut_points)) + '\n']

                        # then do if statements to check if angles above/below threshold
                        angles_each_rep_right.extend((np.array(right_uf_points), np.array(right_ut_points)))
                        # erase lists
                        right_uf_points, right_ut_points = [], []

            count_left = count_angles_between_two_points(left_extremas[-1:][0], ut_angles1[-1:], ut_angles1)
            count_right = count_angles_between_two_points(right_extremas[-1:][0], ut_angles2[-1:], ut_angles2)
            print('angles between left last extrema and angle: ' + str(count_left))
            print('angles between right last extrema and angle: ' + str(count_right))

        if count_left > 20:
            # Last rep analysis
            left_rep_count += 1
            left_reps[left_rep_count] = [
                'Left upper arm - left forearm -> Minimum Angle:' + str(min(left_uf_points)),
                'Left upper arm - left forearm -> Maximum Angle:' + str(max(left_uf_points)),
                'Left upper arm - trunk -> Maximum Angle: ' + str(max(left_ut_points)),
                'Left upper arm - trunk -> Minimum Angle: ' + str(min(left_ut_points)) + '\n']

            # then do if statements to check if angles above/below threshold
            angles_each_rep_left.extend((np.array(left_uf_points), np.array(left_ut_points)))

        if count_right > 20:
            right_rep_count += 1
            right_reps[right_rep_count] = [
                'Right upper arm - left forearm -> Minimum Angle:' + str(min(right_uf_points)),
                'Right upper arm - left forearm -> Maximum Angle:' + str(max(right_uf_points)),
                'Right upper arm - trunk -> Maximum Angle: ' + str(max(right_ut_points)),
                'Right upper arm - trunk -> Minimum Angle: ' + str(min(right_ut_points)) + '\n']

            # then do if statements to check if angles above/below threshold
            angles_each_rep_right.extend((np.array(right_uf_points), np.array(right_ut_points)))

        all_reps = {}
        print('left rep count: ' + str(left_rep_count))
        print('right rep count: ' + str(right_rep_count))
        # Combine dicts for all reps
        if left_rep_count == right_rep_count:
            ds = [left_reps, right_reps]
            for key in left_reps.keys():
                all_reps[key] = tuple(d[key] for d in ds)

            if string == 'dataset':
                return angles_each_rep_left, angles_each_rep_right
            elif string == 'analysis':
                print('Number of reps performed: ' + str(left_rep_count))
                for k, v in all_reps.items():
                    print('\n' + 'Repetition: ' + str(k) + '\n')
                    for s in v:
                        for m in s:
                            print(str(m))
        else:
            print("Error: Rep counts for left and right arms are not equal")
