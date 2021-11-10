from copy import copy
from typing import Optional, Callable

import numpy as np
from scipy import spatial

from r2_lidar_icp.point_cloud import PointCloud
from r2_lidar_icp.sampling import l2_norm, furthest_point_sampling
from r2_lidar_icp.utils import sorted_eig


def no_preprocessing(point_cloud: PointCloud) -> PointCloud:
    return copy(point_cloud)


def random_decimate(point_cloud: PointCloud, percent_keep: float) -> PointCloud:
    pc = copy(point_cloud)
    prob = np.random.uniform(size=pc.features.shape[1])
    mask = (prob < percent_keep)

    pc.features = pc.features[:, mask]
    pc.descriptors = pc.descriptors[:, mask]

    return pc


def furthest_point_sampling_decimate(point_cloud: PointCloud,
                                     nb_keep: int,
                                     initial_idx: Optional[int] = None,
                                     metric: Callable[[np.ndarray, np.ndarray], float] = l2_norm,
                                     skip_initial: bool = False) -> PointCloud:
    pc = copy(point_cloud)
    indices, _ = furthest_point_sampling(pc.features, nb_keep, initial_idx, metric, skip_initial)

    pc.features = pc.features[:, indices]
    pc.descriptors = pc.descriptors[:, indices]

    return pc


# TODO add more descriptors
def make_descriptors(point_cloud: PointCloud, k_nn: int, compute_normals: bool = True) -> PointCloud:
    pc = copy(point_cloud)

    point_dim = pc.features.shape[0] - 1  # exclude homogeneous coordinate

    # compute knn using KDTree
    tree = spatial.KDTree(pc.features.T)
    dist, indices = tree.query(pc.features.T, k=k_nn)

    normals = np.zeros([point_dim, pc.features.shape[1]])

    for i, nn_i in enumerate(indices):
        neighbors = pc.features[:point_dim, nn_i]  # TODO filter points that are too far away
        mu = np.mean(neighbors, axis=1)
        errors = (neighbors.T - mu).T
        cov = 1 / k_nn * (errors @ errors.T)
        eigen_values, eigen_vectors = sorted_eig(cov)
        normals[:, i] = eigen_vectors[:, 0]  # smallest eigen vector

    pc.descriptors = np.zeros((0, pc.features.shape[1]))

    if compute_normals:
        pc.descriptors = np.concatenate([pc.descriptors, normals])

    return pc
