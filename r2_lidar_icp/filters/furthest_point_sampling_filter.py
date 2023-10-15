from typing import Optional, Callable, Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.point_cloud import PointCloud
from r2_lidar_icp.sampling.furthest_point_sampling import l2_norm, furthest_point_sampling


class FurthestPointSamplingFilter(Filter):
    def __init__(self,
                 k: int,
                 initial_idx: Optional[int] = None,
                 metric: Callable[[np.ndarray, np.ndarray], float] = l2_norm,
                 skip_initial: bool = False):
        """
        Filter points using furthest point sampling.
        Will subsample the point cloud to the given number of points, but will sample the points such that they are
        as far apart as possible.
        :param k: Number of points to keep.
        :param initial_idx: Index of the first point to keep. If None, will be chosen randomly.
        :param metric: Metric to use for computing distances between points.
        :param skip_initial: Skip the first furthest point, stabilizes the output
        """
        self.k = k
        self.initial_idx = initial_idx
        self.metric = metric
        self.skip_initial = skip_initial

    def _compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        mask = np.full(pc.features.shape[1], False)
        _, indices = furthest_point_sampling(pc.features, self.k, self.initial_idx, self.metric, self.skip_initial)
        mask[indices] = True
        return mask
