import pickle
from copy import copy
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial

from r2_lidar_icp.draw_utils import IcpInspector
from r2_lidar_icp.point_cloud import PointCloud
from r2_lidar_icp.preprocessing import make_normal_descriptors
from r2_lidar_icp.utils import rigid_transformation


def matching(P, Q):
    """
    Associate two point clouds and produce an error per matching points
    :param P: a 2D point cloud in homogeneous coordinates (3 x n), where n is the number of points.
    :param Q: a 2D point cloud in homogeneous coordinates (3 x m), where m is the number of points.
    :return: an array containing the index of Q matching all points in P.
             The size of the array is n, where n is the number
             of points in P.
    """
    tree = spatial.KDTree(Q.features.T)
    dist, indices = tree.query(P.features.T, k=1)

    return indices


def outlier_filter(P, Q, indices, tau: Optional[float] = None):
    """
    Reduce the impact of outlier.
    :param P: a 2D point cloud in homogeneous coordinates (3 x n), where n is the number of points.
    :param Q: a 2D point cloud in homogeneous coordinates (3 x m), where m is the number of points.
    :param indices: an array representing the misalignment between two point clouds.
    :param tau: distance threshold to filter out outliers
    :return: (distances, P_mask, Q_indices), where P_mask is a mask indicating which points of P are kept,
            and Q_indices is the filtered indices array
    """
    tree = spatial.KDTree(Q.features.T)
    dist, match = tree.query(P.features.T, k=1)

    errors = Q.features[:, match] - P.features
    dist_err = np.linalg.norm(errors, axis=0)

    if tau is not None:
        mask = (dist_err < tau)
    else:
        mask = np.ones_like(dist_err, dtype=int)

    return dist, mask, indices[mask]


def error_minimizer(P, Q, indices, P_mask):
    """
    Minimize an array of errors to produce a rigid transformation.
    :param P: a 2D point cloud in homogeneous coordinates (3 x n), where n is the number of points.
    :param Q: a 2D point cloud in homogeneous coordinates (3 x m), where m is the number of points.
    :param indices: an array representing the misalignment between two point clouds.
    :param P_mask: points to keep in P
    :return: a 2D rigid transformation matrix that moves P to Q
    """
    assert P.features.shape[0] == 3, "only support 2D points in homogenous coords"
    assert P.descriptors.shape[0] == 2, "descriptor must be normal"

    kept_P_features = P.features[:, P_mask]
    nb_pts = kept_P_features.shape[1]

    errors = Q.features[:2, indices] - kept_P_features[:2]
    h = np.empty(nb_pts)
    G = np.empty((3, nb_pts))

    for i in range(nb_pts):
        q_id = indices[i]
        n = Q.descriptors[:, q_id]
        p = kept_P_features[:, i]
        e = errors[:, i]
        h[i] = np.dot(e, n)
        cross = p[0] * n[1] - p[1] * n[0]  # pseudo-cross product in 2D
        G[0:2, i] = n
        G[2, i] = cross

    x = np.linalg.solve(G @ G.T, G @ h)  # this gives: [x, y, theta]
    return rigid_transformation(x)


# TODO add tolerance parameter
# TODO try torch differentiable icp
def icp(P: PointCloud,
        Q: PointCloud,
        nb_iter: int,
        init_pose: Optional[np.ndarray]=None,
        tau_filter: Optional[float]=None,
        min_error_delta: Optional[float]=None,
        inspect: Optional[IcpInspector] = None) -> np.ndarray:
    """
    Point-to-Plane ICP algorithm
    :param P: source point cloud
    :param Q: destination point cloud
    :param nb_iter: maximum number of iteration of the ICP algorithm
    :param init_pose: initial guess on the transformation from P to Q
    :param tau_filter: filter out match that are more than `tau_filter` units from each other
    :param min_error_delta: Stop when the error stops improving by more than `min_error_delta`
    :param inspect: IcpInspector to plot
    :return: Transformation matrix from P to Q
    """
    # initial guess
    if init_pose is not None:
        T = np.copy(init_pose)
    else:
        T = np.eye(3)

    # preprocessing
    P = make_normal_descriptors(P, 20)
    Q = make_normal_descriptors(Q, 20)

    P_prime = copy(P)

    # iterative optimization
    previous_error = 0
    for i in range(nb_iter):
        # move our reading point cloud
        P_prime.features = T @ P.features

        indices = matching(P_prime, Q)
        distances, P_mask, indices = outlier_filter(P_prime, Q, indices, tau_filter)
        T_iter = error_minimizer(P_prime, Q, indices, P_mask)

        # check error
        if min_error_delta is not None:
            mean_error = np.average(distances)
            if abs(previous_error - mean_error) < min_error_delta:
                break
            previous_error = mean_error

        # for plotting later
        if inspect is not None:
            if i == 0:
                inspect.__init__(P_prime, Q, T, indices)
            else:
                inspect.append(P_prime, T, indices)

        # chain transformations
        T = T_iter @ T

    # for plotting later
    if inspect is not None:
        # one last time to apply the last transformation
        P_prime.features = T @ P.features
        indices = matching(P_prime, Q)
        inspect.append(P_prime, T, indices)

    return T


if __name__ == '__main__':
    from r2_lidar_icp.draw_utils import draw_point_clouds

    # generating the reading point cloud
    # angle_p = np.random.uniform(-0.1, 0.1)
    # P = PointCloud()
    # P.features = build_room([1.2, 2.], [2.2, 1.5], angle=angle_p, nb_pts=390)
    P = PointCloud.from_scan(pickle.load(open('data/pi/test1/00000.pkl', 'rb')))

    # generating the reference point cloud
    # angle_q = np.random.uniform(-0.1, 0.1)
    # Q = PointCloud()
    # Q.features = build_room([1.8, 2.], [2.8, 2.2], angle=angle_q, nb_pts=450)
    Q = PointCloud.from_scan(pickle.load(open('data/pi/test1/00050.pkl', 'rb')))

    # an inspector to plot results
    inspector = IcpInspector()

    # calling your iterative closest point algorithm
    T = icp(P, Q, nb_iter=5, tau_filter=100, inspect=inspector)

    # ------------------------------------
    # plotting results
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    ax = axs[0]
    ax.set_title("Before registration")
    draw_point_clouds(ax, P=inspector.P[0].features, Q=inspector.Q.features)

    ax = axs[1]
    ax.set_title("After registration")
    draw_point_clouds(ax, P=inspector.P[-1].features, Q=inspector.Q.features, normals_Q=inspector.Q.descriptors, T=T)

    fig.tight_layout()
    fig.show()
