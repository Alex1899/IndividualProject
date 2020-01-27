class FramePose:
    def __init__(self, keypoints):
        self.keypoints = keypoints
        joints = ['NOSE', 'NECK', 'RSHOULDER', 'RELBOW', 'RWRIST', 'LSHOULDER', 'LELBOW', 'LWRIST', 'MIDHIP',
                  'RHIP', 'RKNEE', 'LHIP', 'LKNEE', 'LANKLE', 'REYE', 'REAR', 'LEAR', 'LBIGTOE', 'LSMALLTOE',
                  'LHEEL', 'RBIGTOE', 'RSMALLTOE', 'RHEEL']
        zipped = zip(joints, keypoints)

        # Map each joint to corresponding keypoint (x,y,c)
        for joint, points in zipped:
            setattr(self, joint, Joint(points))


class Joint:
    def __index__(self, keypoints):
        self.x = keypoints[0]
        self.y = keypoints[1]
        self.c = keypoints[2]
        self.exists = self.c != 0





