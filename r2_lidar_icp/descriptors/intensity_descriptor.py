from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.point_cloud import PointCloud


class IntensityDescriptor(Descriptor):
    """
    Intensity of the return signal.
    """
    name = 'IntensityDescriptor'

    def compute_descriptor(self, pc: PointCloud):
        ...
