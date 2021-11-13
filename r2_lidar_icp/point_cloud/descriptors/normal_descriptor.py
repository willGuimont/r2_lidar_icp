from r2_lidar_icp.point_cloud.descriptors.descriptor import Descriptor
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class NormalDescriptor(Descriptor):
    def __init__(self, knn: int):
        """
        Estimate the normal using the `knn` neighbors
        :param knn: number of neighbors to estimate the normal from
        """
        super().__init__()

    def compute_descriptor(self, pc: PointCloud):
        ...