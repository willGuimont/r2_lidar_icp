import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.matchers.matcher import MatcherType
from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from r2_lidar_icp.utils.utils import sorted_eig


class NormalDescriptor(Descriptor):
    name = 'NormalDescriptor'

    def __init__(self, knn: int, matcher_type: MatcherType):
        """
        Estimate the normal using the `knn` neighbors
        :param knn: number of neighbors to estimate the normal from
        """
        super().__init__()
        self.knn = knn
        self.matcher_type = matcher_type

    def compute_descriptor(self, pc: PointCloud):
        point_dim = pc.features.shape[0] - 1  # exclude homogeneous coordinate

        matcher = self.matcher_type.make_matcher(pc)
        dist, indices = matcher.query(pc, self.knn)

        normals = np.zeros((point_dim, pc.features.shape[1]))

        for i, nn_i in enumerate(indices):
            # TODO filter points that are too far away
            neighbors = pc.features[:point_dim, nn_i]
            mu = np.mean(neighbors, axis=1)
            errors = (neighbors.T - mu).T
            cov = 1 / self.knn * (errors @ errors.T)
            eigen_values, eigen_vectors = sorted_eig(cov)
            normals[:, i] = eigen_vectors[:, 0]  # smallest eigen vector

        pc.add_descriptor(self, normals)
