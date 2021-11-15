from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class MatchDescriptor(Descriptor):
    name = 'MatchDescriptor'

    def __init__(self):
        super().__init__()

    def compute_descriptor(self, pc: PointCloud):
        ...
