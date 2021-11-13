from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from samplers.sampler import Sampler


class FurthestPointSampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, pc: PointCloud):
        ...