from typing import Type

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.matchers.matcher import Matcher
from r2_lidar_icp.point_cloud import PointCloud
from r2_lidar_icp.utils.utils import sorted_eig


class NormalDescriptor(Descriptor):
    name = 'NormalDescriptor'

    def __init__(self, knn: int, matcher_cls: Type[Matcher]):
        """
        Approximate the normal at each point.
        :param knn: Number of nearest neighbors to use.
        :param matcher_cls: Matcher to use to find nearest neighbors.
        """
        self.knn = knn
        self.matcher_cls = matcher_cls

    def compute_descriptor(self, pc: PointCloud):
        point_dim = pc.dim
        num_points = pc.num_points

        matcher = self.matcher_cls.make_matcher(pc)
        dist, indices = matcher.query(pc, self.knn)

        normals = np.zeros((point_dim, num_points))

        for i, nn_i in enumerate(indices):
            neighbors = pc.features[:point_dim, nn_i]
            mu = np.mean(neighbors, axis=1)
            errors = (neighbors.T - mu).T
            cov = 1 / self.knn * (errors @ errors.T)
            eigen_values, eigen_vectors = sorted_eig(cov)
            normals[:, i] = eigen_vectors[:, 0]  # smallest eigen vector

        pc.add_descriptor(self, normals)
