from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.descriptors.normal_descriptor import NormalDescriptor
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.minimizer.minimizer import Minimizer
from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from r2_lidar_icp.utils.utils import rigid_transformation


class PointToPlaneMinimizer(Minimizer):
    # https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
    def find_transformation(self,
                            reference: PointCloud,
                            reading: PointCloud,
                            matches: Matches,
                            descriptors: Dict[str, Descriptor]):
        assert reference.features.shape[0] == 3, "only support 2D points"

        distances, indices = matches.distances, matches.indices
        dim = reading.dim
        nb_points = reading.features.shape[1]

        ref_normals = reference.get_descriptor(NormalDescriptor.name, descriptors)

        errors = reference.features[:2, indices] - reading.features[:2, :]
        h = np.empty(nb_points)
        G = np.empty((dim + 1, nb_points))

        for i in range(nb_points):
            q_id = indices[i]
            n = ref_normals[:, q_id]
            p = reading.features[:, i]
            e = errors[:, i]
            h[i] = np.dot(e, n)
            cross = self._cross_product(p, n, dim)
            G[:dim, i] = n
            G[dim, i] = cross
        x = np.linalg.solve(G @ G.T, G @ h)  # this gives: [x, y, theta]
        return rigid_transformation(x)

    @staticmethod
    def _cross_product(p, n, dim):
        if dim == 2:
            return p[0] * n[1] - p[1] * n[0]  # pseudo-cross product in 2D
        elif dim == 3:
            return np.cross(p, n)
        else:
            raise ValueError(f'invalid dimension {dim}')
