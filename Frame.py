class FramePose:
    def __init__(self, keypoints):
        self.keypoints = keypoints
        list_joints = ['NOSE', 'NECK', 'RSHOULDER', 'RELBOW', 'RWRIST', 'LSHOULDER', 'LELBOW', 'LWRIST', 'MIDHIP',
                  'RHIP', 'RKNEE', 'LHIP', 'LKNEE', 'LANKLE', 'REYE', 'REAR', 'LEAR', 'LBIGTOE', 'LSMALLTOE',
                  'LHEEL', 'RBIGTOE', 'RSMALLTOE', 'RHEEL']
        zipped = zip(list_joints, keypoints)

        # list of tuples (joint_name, exists ?, [x,y,c])
        self.joint_keypoints = {}
        # Map each joint to corresponding keypoint (x,y,c)
        for joint_name, points in zipped:
            self.joint_keypoints[joint_name] = points




