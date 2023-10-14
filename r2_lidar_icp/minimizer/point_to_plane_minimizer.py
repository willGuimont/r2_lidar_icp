from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.descriptors.normal_descriptor import NormalDescriptor
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.minimizer.minimizer import Minimizer
from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from r2_lidar_icp.utils.utils import rigid_transformation


class PointToPlaneMinimizer(Minimizer):
    """
    Point to plane minimizer.
    Inspired by https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
    """

    def find_transformation(self, point_cloud: PointCloud, reference: PointCloud, matches: Matches,
                            descriptors: Dict[str, Descriptor]):
        # TODO add support for 3D points
        assert reference.features.shape[0] == 3, "only support 2D points"

        distances, indices = matches.distances, matches.indices
        dim = point_cloud.dim
        nb_points = point_cloud.features.shape[1]

        ref_normals = reference.get_descriptor(NormalDescriptor.name, descriptors)

        errors = reference.features[:2, indices] - point_cloud.features[:2, :]
        h = np.empty(nb_points)
        G = np.empty((dim + 1, nb_points))

        for i in range(nb_points):
            q_id = indices[i]
            n = ref_normals[:, q_id]
            p = point_cloud.features[:, i]
            e = errors[:, i]
            h[i] = np.dot(e, n)
            cross = self._cross_product(p, n, dim)
            G[:dim, i] = n
            G[dim, i] = cross
        x = np.linalg.solve(G @ G.T, G @ h)  # this gives: [x, y, theta]
        # TODO use transformation matrix instead?
        return rigid_transformation(x)

    @staticmethod
    def _cross_product(p, n, dim):
        """
        Wrapper to compute the cross product between a 2D vector and a 3D vector
        :param p: first vector
        :param n: second vector
        :param dim: dimension of the vectors
        :return: cross product between p and n
        """
        if dim == 2:
            return p[0] * n[1] - p[1] * n[0]  # pseudo-cross product in 2D
        elif dim == 3:
            return np.cross(p, n)
        else:
            raise ValueError(f'invalid dimension {dim}')
