from copy import copy
from typing import Dict

import numpy as np

from r2_lidar_icp.utils.utils import point_to_homogeneous


class PointCloud:
    def __init__(self, features: np.ndarray):
        self.features = features
        self.descriptors = dict()

    def get_descriptor(self, descriptor_name: str, descriptors: Dict[str, 'Descriptor']):
        if descriptor_name not in self.descriptors:
            if descriptor_name not in descriptors:
                raise RuntimeError(f'Descriptor {descriptor_name} was not computed and is not in descriptors')
            descriptor = descriptors[descriptor_name]
            descriptor.compute_descriptor(self)
        return self.descriptors[descriptor_name]

    def add_descriptor(self, descriptor: 'Descriptor', value):
        self.descriptors[descriptor.name] = value

    def apply_mask(self, mask: np.ndarray):
        self.features = self.features[:, mask]
        for k, v in self.descriptors.items():
            self.descriptors[k] = v[:, mask]

    @property
    def dim(self):
        return self.features.shape[0] - 1

    @property
    def homogeneous_dim(self):
        return self.features.shape[0]

    @property
    def num_points(self):
        return self.features.shape[1]

    def __copy__(self):
        pc = PointCloud(copy(self.features))

        for k, v in self.descriptors.items():
            pc.descriptors[k] = copy(v)

        return pc

    @classmethod
    def from_scan(cls, scan) -> 'PointCloud':
        scan = np.array(scan)

        qualities, angles, distances = scan[:, 0], scan[:, 1], scan[:, 2]
        angles = np.deg2rad(angles)

        xs = np.cos(angles) * distances
        ys = np.sin(angles) * distances

        features = np.stack([xs, ys])
        features = point_to_homogeneous(features)

        pc = PointCloud(features)

        return pc
