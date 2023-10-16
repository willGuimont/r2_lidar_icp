from abc import ABC, abstractmethod

from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.point_cloud import PointCloud


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
        :return: Mask to apply to the matches (n, 1).
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
