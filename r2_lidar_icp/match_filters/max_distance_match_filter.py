from r2_lidar_icp.match_filters.match_filter import MatchFilter
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud import PointCloud


class MaxDistanceMatchFilter(MatchFilter):
    def __init__(self, max_distance: float):
        """
        Filter matches based on distance.
        Matches with a distance greater than max_distance will be removed.
        :param max_distance: Maximum distance to keep.
        """
        self.max_distance = max_distance

    def _compute_mask(self, pc: PointCloud, matches: Matches):
        return matches.best_distances < self.max_distance
