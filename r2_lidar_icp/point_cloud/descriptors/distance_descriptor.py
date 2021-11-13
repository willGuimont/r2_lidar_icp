from r2_lidar_icp.point_cloud.descriptors.descriptor import Descriptor
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class DistanceDescriptor(Descriptor):
    def __init__(self):
        """
        Adds the distance to the closest neighbor in the map of each point
        """
        super().__init__()

    def compute_descriptor(self, pc: PointCloud):
        ...