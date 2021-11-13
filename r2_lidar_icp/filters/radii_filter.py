from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class RadiiFilter(Filter):
    def __init__(self):
        """
        Filters out points that are not in the specified
        """
        super().__init__()

    def keep_indices(self, pc: PointCloud):
        ...
