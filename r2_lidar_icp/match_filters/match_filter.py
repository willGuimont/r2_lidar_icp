from abc import ABC, abstractmethod

from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud import PointCloud


# TODO Lowe's ratio test dist_1 / dist_2 < 0.8 or 0.6
class MatchFilter(ABC):
    """
Base class for match filters. Match filters will remove matches from the point cloud based on some criteria.
    """

    @abstractmethod
    def _compute_mask(self, pc: PointCloud, matches: Matches):
        """
        Compute the mask to apply to the matches.
        :param pc: Point cloud.
        :param matches: Matches to compute the mask for.
        :return: Mask to apply to the matches (boolean array with same length as matches).
        """

    def filter_matches(self, pc: PointCloud, matches: Matches):
        """
        Apply the filter to the matches.
        :param pc: Point cloud.
        :param matches: Matches to apply the filter to.
        :return: None
        """
        mask = self._compute_mask(pc, matches)
        matches.apply_mask(mask)
