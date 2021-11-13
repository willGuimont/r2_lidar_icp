import numpy as np


def sorted_eig(A):
    eigen_values, eigen_vectors = np.linalg.eig(A)
    idx = np.argsort(eigen_values)
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return eigen_values, eigen_vectors


def point_to_homogeneous(pc):
    if pc.shape[0] == 3:
        return np.copy(pc.T)
    elif pc.shape[0] == 2:
        cp = np.copy(pc)
        return np.concatenate((cp, np.ones((1, cp.shape[1]))), axis=0)
    else:
        raise ValueError(f'{pc.shape} is an invalide shape, expected Nx3 or Nx4')


def homogeneous_to_points(pc):
    assert pc.shape[0] == 3, "only work with 2d points"
    return pc[:2, :] / pc[2, :]


def rigid_transformation(params):
    """
    Returns a rigid transformation matrix
    :params: numpy array, params[0]=tx, params[1]=ty, params[2]=theta
    :returns: LaTeX bmatrix as a string
    """
    return np.array([[np.cos(params[2]), -np.sin(params[2]), params[0]],
                     [np.sin(params[2]), np.cos(params[2]), params[1]],
                     [0, 0, 1]])


def mode_beta(param):
    alpha = param[0]
    beta = param[1]
    return (alpha - 1) / (alpha + beta - 2)


def build_room(param_v, param_h, angle=0., wall_thickness=0.01, nb_pts=400):
    nb_pts = int(nb_pts / 4)

    sensor_center = np.ones(3)

    wall_top = np.ones([3, nb_pts])
    wall_top[0] = np.random.beta(param_v[0], param_v[1], nb_pts)
    wall_top[1] = np.random.uniform(-wall_thickness / 2., wall_thickness / 2., nb_pts) + 1.

    wall_bottom = np.ones([3, nb_pts])
    wall_bottom[0] = np.random.beta(param_v[0], param_v[1], nb_pts)
    wall_bottom[1] = np.random.uniform(-wall_thickness / 2., wall_thickness / 2., nb_pts)
    sensor_center[0] = mode_beta(param_v)

    wall_left = np.ones([3, nb_pts])
    wall_left[1] = np.random.beta(param_h[0], param_h[1], nb_pts)
    wall_left[0] = np.random.uniform(-wall_thickness / 2., wall_thickness / 2., nb_pts)

    wall_right = np.ones([3, nb_pts])
    wall_right[1] = np.random.beta(param_h[0], param_h[1], nb_pts)
    wall_right[0] = np.random.uniform(-wall_thickness / 2., wall_thickness / 2., nb_pts) + 1.

    sensor_center[1] = mode_beta(param_h)

    T = rigid_transformation([-sensor_center[0], -sensor_center[1], angle])
    P = np.hstack([wall_bottom, wall_top, wall_left, wall_right])

    return T @ P


def build_parallelepiped(P):
    assert (P.shape[0] == 3), "Wrong number of dimensions"
    assert (P.shape[1] == 8), "Wrong number of points"
    return [[P[:, 0], P[:, 1], P[:, 2], P[:, 3]],
            [P[:, 4], P[:, 5], P[:, 6], P[:, 7]],
            [P[:, 0], P[:, 1], P[:, 5], P[:, 4]],
            [P[:, 2], P[:, 3], P[:, 7], P[:, 6]],
            [P[:, 1], P[:, 2], P[:, 6], P[:, 5]],
            [P[:, 4], P[:, 7], P[:, 3], P[:, 0]]]
