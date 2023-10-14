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
    :param pc: point cloud
    :return: homogeneous coordinates
    """
    if pc.shape[0] == 3:
        return np.copy(pc.T)
    elif pc.shape[0] == 2:
        cp = np.copy(pc)
        return np.concatenate((cp, np.ones((1, cp.shape[1]))), axis=0)
    else:
        raise ValueError(f'{pc.shape} is an invalide shape, expected Nx3 or Nx4')


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
