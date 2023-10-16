import numpy as np

from r2_lidar_icp.match_filters.match_filter import MatchFilter
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud import PointCloud


class IdentityMatchFilter(MatchFilter):
    """
    Identity match filter. Does not filter any matches.
    """

    def _compute_mask(self, pc: PointCloud, matches: Matches):
        return np.full((pc.features.shape[1], 1), True)
