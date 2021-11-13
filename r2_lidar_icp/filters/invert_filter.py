from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class InvertFilter(Filter):
    def __init__(self, filter: Filter):
        """
        Inverts a filter
        :param filter: filter to invert
        """
        super().__init__()

    def keep_indices(self, pc: PointCloud):
        ...
