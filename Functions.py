import numpy as np
import math


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

    print('Right side: '+str(sum(rside_joints)))
    print('Left  side: '+str(sum(lside_joints)))
    
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


# Could modify this function to use across other exercise classes
# local minima points are minimum angles in each rep, no need to calc again
def analyse_each_rep(string, extremas1, uf_angles1a, ut_angles2a, tk_angles3a=None, extremas2=None, uf_angles1b=None, ut_angles2b=None):
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

        for (uf_p, ut_p, tk_p) in zip(uf_angles1a, ut_angles2a, tk_angles3a):
            if uf_p not in extremas1:
                uf_points.append(uf_p)
                ut_points.append(ut_p)
                tk_points.append(tk_p)

            else:
                if max_counter == extremas1.size:
                    uf_points.append(uf_p)
                    ut_points.append(ut_p)
                    tk_points.append(tk_p)
                else:
                    # fix for duplicates
                    index = np.where(extremas1 == uf_points)
                    if index[0]:
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

        # Last rep analysis

        all_reps[rep_count + 1] = [
            "Minimum angle between upper arm and forearm: " + str(min(uf_points)),
            "Maximum angle between upper arm and trunk: " + str(max(ut_points)),
            "Minimum angle between trunk and knee: " + str(min(tk_points))]

        angles_each_rep.extend((np.array(uf_points), np.array(ut_points), np.array(tk_points)))
        if string == 'dataset':
            return angles_each_rep
        elif string == 'analysis':
            for k, v in all_reps.items():
                print('Repetition: ' + str(k))
                for s in v:
                    print(s + '\n')

    else:
        list_maxes_left, list_maxes_right, left_uf_points, right_uf_points, left_ut_points, right_ut_points = [], [], [], [], [], []
        angles_each_rep_left, angles_each_rep_right = [], []
        left_rep_count, right_rep_count = 0, 0
        left_max_counter, right_max_counter = 0, 0

        left_reps, right_reps = {}, {}
        left_extremas = extremas1
        right_extremas = extremas2

        if extremas1.size != extremas2.size:
            print("Left and Right maxima points not equal")

        for (left_uf_p, right_uf_p, left_ut_p, right_ut_p) in zip(uf_angles1a, uf_angles1b, ut_angles2a, ut_angles2b):
            while left_rep_count != left_extremas.size and right_rep_count != right_extremas.size:
                if left_uf_p not in left_extremas:
                    left_uf_points.append(left_uf_p)
                    left_ut_points.append(left_ut_p)
                else:
                    """
                    if left_max_counter == left_extremas.size:
                        left_uf_points.append(left_uf_p)
                        left_ut_points.append(left_ut_p)
                    """
                    # else:
                    # fix for duplicates
                    # print("left max: " + str(left_uf_p))
                    index = np.where(left_extremas == left_uf_p)
                    print("Left max indexes:  " + str(index))

                    # print(np.take(extremas1, index[0]))
                    left_extremas = np.delete(left_extremas, index[0][0])

                    left_uf_points.append(left_uf_p)
                    left_rep_count += 1
                    left_max_counter += 1
                    left_reps[left_rep_count] = [
                            'Left upper arm - left forearm -> Minimum Angle:' + str(min(left_uf_points)),
                            'Left upper arm - left forearm -> Maximum Angle:' + str(max(left_uf_points)),
                            'Left upper arm - trunk -> Maximum Angle: ' + str(max(left_ut_points)),
                            'Left upper arm - trunk -> Minimum Angle: ' + str(min(left_ut_points))]

                    # then do if statements to check if angles above/below threshold
                    angles_each_rep_left.extend((np.array(left_uf_points), np.array(left_ut_points)))
                    # erase lists
                    left_uf_points, left_ut_points = [], []

                if right_uf_p not in right_extremas:
                    right_uf_points.append(right_uf_p)
                    right_ut_points.append(right_ut_p)
                else:
                    """
                    if right_max_counter == right_extremas.size:
                        right_uf_points.append(right_uf_p)
                        right_ut_points.append(right_ut_p)
                    """
                    #else:
                    # fix for duplicates
                    # need to fix it ...!!!!
                    # print("right max: " + str(right_uf_p))
                    index = np.where(right_extremas == right_uf_p)

                    print("Right max indexes:  " + str(index))
                    # print(index[0])
                    # print(np.take(extremas2, index[0]))
                    right_extremas = np.delete(right_extremas, index[0][0])

                    right_uf_points.append(right_uf_p)
                    right_max_counter += 1
                    right_rep_count += 1
                    right_reps[right_rep_count] = [
                            'Right upper arm - left forearm -> Minimum Angle:' + str(min(right_uf_points)),
                            'Right upper arm - left forearm -> Maximum Angle:' + str(max(right_uf_points)),
                            'Right upper arm - trunk -> Maximum Angle: ' + str(max(right_ut_points)),
                            'Right upper arm - trunk -> Minimum Angle: ' + str(min(right_ut_points))]

                    # then do if statements to check if angles above/below threshold
                    angles_each_rep_right.extend((np.array(right_uf_points), np.array(right_ut_points)))
                    # erase lists
                    right_uf_points, right_ut_points = [], []

            if left_max_counter == left_extremas.size:
                left_uf_points.append(left_uf_p)
                left_ut_points.append(left_ut_p)

            if right_max_counter == right_extremas.size:
                right_uf_points.append(right_uf_p)
                right_ut_points.append(right_ut_p)

            #

        # Last rep analysis
        left_rep_count += 1
        left_reps[left_rep_count] = [
            'Left upper arm - left forearm -> Minimum Angle:' + str(min(left_uf_points)),
            'Left upper arm - left forearm -> Maximum Angle:' + str(max(left_uf_points)),
            'Left upper arm - trunk -> Maximum Angle: ' + str(max(left_ut_points)),
            'Left upper arm - trunk -> Minimum Angle: ' + str(min(left_ut_points))]

        # then do if statements to check if angles above/below threshold
        angles_each_rep_left.extend((np.array(left_uf_points), np.array(left_ut_points)))

        right_rep_count += 1
        right_reps[right_rep_count] = [
            'Right upper arm - left forearm -> Minimum Angle:' + str(min(right_uf_points)),
            'Right upper arm - left forearm -> Maximum Angle:' + str(max(right_uf_points)),
            'Right upper arm - trunk -> Maximum Angle: ' + str(max(right_ut_points)),
            'Right upper arm - trunk -> Minimum Angle: ' + str(min(right_ut_points))]

        # then do if statements to check if angles above/below threshold
        angles_each_rep_right.extend((np.array(right_uf_points), np.array(right_ut_points)))

        all_reps = {}

        print(left_rep_count)
        print(right_rep_count)

        # Combine dicts for all reps
        if left_rep_count == right_rep_count:
            ds = [left_reps, right_reps]
            for key in left_reps.keys():
                print
                all_reps[key] = tuple(d[key] for d in ds)
        else:
            print(left_rep_count)
            print(right_rep_count)
            print("Rep counts for left and right are not equal")

        if string == 'dataset':
            return angles_each_rep_left, angles_each_rep_right
        elif string == 'analysis':
            for k, v in all_reps.items():
                print('\n'+'Repetition: ' + str(k) + '\n')
                for s in v:
                    for m in s:
                        print(str(m))


