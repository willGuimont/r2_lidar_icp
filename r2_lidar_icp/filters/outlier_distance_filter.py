from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class OutlierDistanceFilter(Filter):
    def __init__(self):
        """
        Filters out outliers
        """
        super().__init__()

    def keep_indices(self, pc: PointCloud):
        ...
