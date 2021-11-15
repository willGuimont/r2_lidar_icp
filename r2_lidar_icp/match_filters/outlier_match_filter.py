from r2_lidar_icp.match_filters.match_filter import MatchFilter
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class OutlierMatchFilter(MatchFilter):
    def __init__(self, max_distance: float):
        self.max_distance = max_distance

    def compute_mask(self, pc: PointCloud, matches: Matches):
        return matches.distances < self.max_distance
