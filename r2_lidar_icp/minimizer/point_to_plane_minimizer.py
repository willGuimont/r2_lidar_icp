from collections.abc import Callable
from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.descriptors.normal_descriptor import NormalDescriptor
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.minimizer.minimizer import Minimizer
from r2_lidar_icp.point_cloud import PointCloud
from r2_lidar_icp.utils.utils import rigid_transformation, pseudo_cross_product


class PointToPlaneMinimizer(Minimizer):
    def __init__(self, weight_function: Callable = None):
        """
        Point to plane minimizer.
        Inspired by https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
        """
        self.weight_function = weight_function

    # TODO add comments to explain the math behind this
    def find_transformation(self, point_cloud: PointCloud, reference: PointCloud, matches: Matches,
                            descriptors: Dict[str, Descriptor]):
        # TODO add support for 3D points
        assert reference.features.shape[0] == 3, 'only support 2D points'

        distances, from_indices, indices = matches.best_distances, matches.from_indices, matches.best_indices
        dim = point_cloud.dim
        nb_matches = matches.num_matches

        if self.weight_function is not None:
            weights = self.weight_function(distances)

        ref_normals = reference.get_descriptor(NormalDescriptor.name, descriptors)

        errors = reference.features[:2, indices] - point_cloud.features[:2, from_indices]
        h = np.empty(nb_matches)
        G = np.empty((dim + 1, nb_matches))

        for i in range(nb_matches):
            q_id = indices[i]
            n = ref_normals[:, q_id]
            p = point_cloud.features[:, i]
            e = errors[:, i]
            h[i] = np.dot(e.T, n)
            cross = pseudo_cross_product(p, n, dim)
            G[:dim, i] = n.squeeze()
            G[dim, i] = cross
            if self.weight_function is not None:
                w = weights[i]
                h[i] *= w
                G[0:2, i] *= w
                G[2, i] *= w
        x = np.linalg.solve(G @ G.T, G @ h)  # this gives: [x, y, theta]
        # TODO use transformation matrix instead?
        return rigid_transformation(x)
