from typing import Optional, Callable, Dict

import numpy as np

from backup.point_cloud import PointCloud
from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.filters.filter import Filter
from r2_lidar_icp.sampling.furthest_point_sampling import l2_norm, furthest_point_sampling


class FurthestPointSamplingFilter(Filter):
    def __init__(self,
                 k: int,
                 initial_idx: Optional[int] = None,
                 metric: Callable[[np.ndarray, np.ndarray], float] = l2_norm,
                 skip_initial: bool = False):
        self.k = k
        self.initial_idx = initial_idx
        self.metric = metric
        self.skip_initial = skip_initial

    def compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        mask = np.full(pc.features.shape[1], False)
        indices, _ = furthest_point_sampling(pc.features, self.k, self.initial_idx, self.metric, self.skip_initial)
        mask[indices] = True
        return mask
