from typing import Dict

import numpy as np

from r2_lidar_icp.descriptors.descriptor import Descriptor
from r2_lidar_icp.matchers.matches import Matches
from r2_lidar_icp.minimizer.minimizer import Minimizer
from r2_lidar_icp.point_cloud import PointCloud


# TODO debug this and validate
class PointToPointMinimizer(Minimizer):
    """
    Point to point minimizer.
    Inspired by https://github.com/norlab-ulaval/glo4001/blob/master/robmob/icp.py
    """

    # TODO add comments to explain the math behind this
    def find_transformation(self, point_cloud: PointCloud, reference: PointCloud, matches: Matches,
                            descriptors: Dict[str, Descriptor]):
        dim = reference.dim
        distances, from_indices, indices = matches.best_distances, matches.from_indices, matches.best_indices

        num_points = indices.shape[0]
        A = np.zeros((num_points, 3))
        B = np.zeros((num_points, 3))
        A[:, :dim] = point_cloud.features[:dim, from_indices].T
        B[:, :dim] = reference.features[:dim, indices].T

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        # homogeneous transformation
        T = np.identity(reference.homogeneous_dim)
        T[0:dim, 0:dim] = R[:dim, :dim]
        T[0:dim, dim] = t[:dim]

        return T
