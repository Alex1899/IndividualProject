import numpy as np
import math


# Detecting sides (left or right) on videos
# can use this method for other exercises
def detect_side(frame_poses):
    side = ''
    rside_joints = [
        pose.joint_keypoints['RSHOULDER'] + pose.joint_keypoints['RELBOW'] + pose.joint_keypoints['RWRIST'] +
        pose.joint_keypoints['RHIP'] + pose.joint_keypoints['RKNEE'] for pose in frame_poses
        if pose.joint_keypoints['RSHOULDER'][2] != 0 and pose.joint_keypoints['RELBOW'][2] != 0 and
        pose.joint_keypoints['RWRIST'][2] != 0 and pose.joint_keypoints['RHIP'][2] != 0 and
        pose.joint_keypoints['RKNEE'][2] != 0]

    lside_joints = [
        pose.joint_keypoints['LSHOULDER'] + pose.joint_keypoints['LELBOW'] + pose.joint_keypoints['LWRIST'] +
        pose.joint_keypoints['LHIP'] + pose.joint_keypoints['LKNEE'] for pose in frame_poses
        if pose.joint_keypoints['LSHOULDER'][2] != 0 and pose.joint_keypoints['LELBOW'][2] != 0 and
        pose.joint_keypoints['LWRIST'][2] != 0 and pose.joint_keypoints['LHIP'][2] != 0 and
        pose.joint_keypoints['LKNEE'][2] != 0]

    # think about the case when they are equal
    rcount, lcount = 0, 0
    for r, l in zip(rside_joints, lside_joints):
        rcount += len(r)
        lcount += len(l)

    if rcount > lcount:
        side = 'right'
        return side
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
def analyse_each_rep(angles1, angles2, angles3, maximas):
    if maximas.size == 0:
        return 0
    list_maxs, uf_points, ut_points, tk_points = [], [], [], []
    rep_count = 0
    max_counter = 0
    for (tk_p, uf_p, ut_p) in zip(angles1, angles2, angles3):
        if uf_p not in maximas:
            uf_points.append(uf_p)
            ut_points.append(ut_p)
            tk_points.append(tk_p)

        else:
            if max_counter == maximas.size:
                uf_points.append(uf_p)
                ut_points.append(ut_p)
                tk_points.append(tk_p)
            elif uf_p in list_maxs:
                continue
            else:
                # fix for duplicates
                list_maxs.append(uf_p)
                rep_count += 1
                max_counter += 1
                print('Repetition: ' + str(rep_count))
                print("Minimum angle between upper arm and forearm: " + str(min(uf_points)))
                print("Maximum angle between upper arm and trunk: " + str(max(ut_points)))
                print("Maximum angle between trunk and knee: " + str(max(tk_points)))
                print('\n')
                # then do if statements to check if angles above/below threshold
                ut_points, tk_points = [], []

    # Last rep analysis
    print('Repetition: ' + str(rep_count + 1))
    print("Minimum angle between upper arm and forearm: " + str(min(uf_points)))
    print("Maximum angle between upper arm and trunk: " + str(max(ut_points)))
    print("Maximum angle between trunk and knee: " + str(max(tk_points)))
    return 1
