from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.point_cloud.point_cloud import PointCloud


class Filter(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def compute_mask(self, pc: PointCloud, descriptors: Dict[str, Descriptor]) -> np.ndarray:
        ...

    def filter(self, pc: PointCloud, descriptors: Dict[str, Descriptor]):
        mask = self.compute_mask(pc, descriptors)
        pc.apply_mask(mask)
