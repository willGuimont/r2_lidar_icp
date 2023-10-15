from typing import Optional

import numpy as np


def sorted_eig(A: np.ndarray):
    """
    Returns eigen values and eigen vectors sorted by eigen values
    TODO what if eigen values are complex or negative?
    :param A: matrix
    :return: eigen values and eigen vectors sorted by eigen values
    """
    eigen_values, eigen_vectors = np.linalg.eig(A)
    idx = np.argsort(eigen_values)
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return eigen_values, eigen_vectors


def point_to_homogeneous(pc: np.ndarray):
    """
    Convert a point cloud to homogeneous coordinates
    :param pc: point cloud (3, n) or (2, n)
    :return: homogeneous coordinates (4, n) or (3, n)
    """
    cp = np.copy(pc)
    return np.concatenate((cp, np.ones((1, cp.shape[1]))), axis=0)


def rigid_transformation(params):
    """
    Returns a rigid transformation matrix
    TODO remove this function
    :params: numpy array, params[0]=tx, params[1]=ty, params[2]=theta
    :returns: LaTeX bmatrix as a string
    """
    assert params.shape[0] == 3, "only support 2D points"
    return np.array([[np.cos(params[2]), -np.sin(params[2]), params[0]],
                     [np.sin(params[2]), np.cos(params[2]), params[1]],
                     [0, 0, 1]])


def line_line_intersection_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns the intersection point between two lines in 2D
    :param p1: First point of the first line
    :param p2: Second point of the first line
    :param p3: First point of the second line
    :param p4: Second point of the second line
    :return: Intersection point between the two lines, None if the lines do not intersect
    """
    denominator = (p4[0] - p3[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p4[1] - p3[1])
    if denominator == 0:
        return None
    t1 = ((p3[1] - p4[1]) * (p1[0] - p3[0]) + (p4[0] - p3[0]) * (p1[1] - p3[1])) / denominator
    t2 = ((p1[1] - p2[1]) * (p1[0] - p3[0]) + (p2[0] - p1[0]) * (p1[1] - p3[1])) / denominator
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return p1 + t1 * (p2 - p1)
    else:
        return None


def l1_norm(x: np.ndarray, y: np.ndarray):
    """
    L1 norm between two vectors
    :param x: First vectors (3, n)
    :param y: Second vectors (3, m)
    :return: L1 norm between x and y
    """
    return np.abs(x - y).sum(axis=0)


def l2_norm(x: np.ndarray, y: np.ndarray):
    """
    L2 norm between two vectors
    :param x: First vectors (3, n)
    :param y: Second vectors (3, m)
    :return: L2 norm between x and y
    """
    return ((x - y) ** 2).sum(axis=0)


def lp_norm(x: np.ndarray, y: np.ndarray, p: float):
    """
    Lp norm between two vectors
    :param x: First vectors (3, n)
    :param y: Second vectors (3, m)
    :param p: p value
    :return: Lp norm between x and y
    """
    return np.power(np.abs(x - y), p).sum(axis=0)
