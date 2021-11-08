import numpy as np
import torch


def scan_to_point_cloud(scan):
    scan = torch.from_numpy(scan)

    qualities, angles, distances = scan[:, 0], scan[:, 1], scan[:, 2]
    angles = torch.deg2rad(angles)

    xs = torch.cos(angles) * distances
    ys = torch.sin(angles) * distances
    zs = torch.zeros_like(xs)

    stack = torch.stack([xs, ys, zs]).T
    return stack


def point_to_homogeneous(pc):
    """
    :param pc: Nx2, Nx3 or Nx4 numpy array
    :return: deep copy of pc in homogeneous coordinates.
    """
    if pc.shape[1] == 4:
        return pc.T
    elif pc.shape[1] == 3:
        return torch.cat((pc.T, torch.ones((1, pc.shape[0]))), dim=0)
    elif pc.shape[1] == 2:
        return torch.cat((pc.T, torch.zeros((1, pc.shape[0])), torch.ones((1, pc.shape[0]))), dim=0)
    else:
        raise ValueError(f'{pc.shape} is an invalide shape, expected Nx3 or Nx4')


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    :param src: Nx3 array of points
    :param dst: Nx3 array of points
    :return: (distance, indices):
            distance: Euclidean distances (errors) of the nearest neighbor
            indices: dst indices of the nearest neighbor
    """
    indices = torch.zeros(src.shape[0], dtype=torch.long)
    distances = torch.zeros(src.shape[0])
    for i, s in enumerate(src):
        distance_mat = torch.linalg.norm(dst[:] - s, axis=1)
        indices[i] = torch.argmin(distance_mat)
        distances[i] = distance_mat[indices[i]]
    return distances, indices


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    :param A: Nx3 numpy array of corresponding 3D points
    :param B: Nx3 numpy array of corresponding 3D points
    :return: (T, R, t)
             T: 4x4 homogeneous transformation matrix
             R: 3x3 rotation matrix
             t: 3x1 column vector
    """
    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = torch.mean(A, axis=0)
    centroid_B = torch.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = AA.T @ BB
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if torch.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # translation
    t = centroid_B.T - R @ centroid_A.T

    # homogeneous transformation
    T = torch.eye(4, dtype=A.dtype)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t


def huber_loss(distances, delta, reduction='mean'):
    def loss_fn(d):
        if d < delta:
            return 0.5 * d ** 2
        else:
            return delta * (d - 0.5 * delta)

    losses = torch.zeros_like(distances)
    for i in range(distances.shape[0]):
        losses[i] = loss_fn(distances[i])

    if reduction == 'mean':
        return torch.mean(losses)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise ValueError(f'invalid reduction: {reduction}')


def transformation_matrix(theta, translation):
    c = torch.cos(theta)
    s = torch.sin(theta)
    x = translation[0]
    y = translation[1]
    T = torch.eye(4, dtype=torch.float64)

    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = x
    T[1, 3] = y

    return T


def icp(A, B, init_pose=None, lr=0.0001, max_iter=100, tolerance=0.001):
    """
    The Iterative Closest Point method
    :param A: Nx3 or Nx4 numpy array of source 3D points
    :param B: Nx3 or Nx4 numpy array of destination 3D point
    :param init_pose: 4x4 homogeneous transformation
    :param max_iter: exit algorithm after max_iterations
    :param tolerance: convergence criteria
    :return: (T, distances)
            T: final homoheneous transformation
            distances: euclidean distances (errors) of the nearest neighbor
    """
    src = point_to_homogeneous(A)
    dst = point_to_homogeneous(B)

    if init_pose is not None:
        src = torch.dot(init_pose, src)

    window = "iter"
    window_size = 500
    cv2.namedWindow(window)

    distances, indices = nearest_neighbor(src[:3, :].T, dst[:3, :].T)
    T, _, _ = best_fit_transform(src[:3, :].T, dst[:3, indices].T)

    prev_error = 0
    distances = None
    src_prim = None
    theta = torch.zeros((1), dtype=src.dtype, requires_grad=True)
    translation = torch.zeros((2), dtype=src.dtype, requires_grad=True)

    with torch.no_grad():
        theta[:] = torch.arccos(T[0, 0])
        translation[:] = T[:2, 3]

    for i in range(max_iter):
        T = transformation_matrix(theta, translation)
        src_prim = T @ src
        distances, indices = nearest_neighbor(src_prim[:3, :].T, dst[:3, :].T)
        error = torch.mean(distances)
        if error < 200:
            error = huber_loss(distances, 1)
        print(error)
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
        # if i != 0 and i % 10 == 0:
        #     lr *= 0.1

        if torch.any(torch.isinf(src_prim)):
            print('wtf')

        img = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        draw_point_cloud(src.detach().numpy(), img, window_size, (255, 255, 0))
        draw_point_cloud(dst.detach().numpy(), img, window_size, (255, 0, 255))
        draw_point_cloud(src_prim.detach().numpy(), img, window_size, (255, 255, 255))
        cv2.imshow(window, img)
        cv2.waitKey(250)

        error.backward()

        with torch.no_grad():
            theta -= lr * theta.grad
            translation -= lr * translation.grad

        theta.grad.zero_()
        translation.grad.zero_()

    T, _, _ = best_fit_transform(A[:, :3], src_prim[:3, :].T.detach())
    return T, distances


if __name__ == '__main__':
    import pickle
    from cv2 import cv2
    from tools.visualize_point_cloud import draw_point_cloud

    scan_0 = pickle.load(open('data/live/8.pkl', 'rb'))
    scan_1 = pickle.load(open('data/live/85.pkl', 'rb'))

    pc0 = scan_to_point_cloud(scan_0)
    pc1 = scan_to_point_cloud(scan_1)

    T, distances = icp(pc0, pc1)
    print(T)

    window = "point_cloud"
    window_size = 500
    cv2.namedWindow(window)

    src = point_to_homogeneous(pc0)
    dst = point_to_homogeneous(pc1)

    img = np.zeros((window_size, window_size, 3), dtype=np.uint8)
    draw_point_cloud(src, img, window_size, (255, 255, 0))
    draw_point_cloud(dst, img, window_size, (255, 0, 255))
    draw_point_cloud(T @ point_to_homogeneous(pc0), img, window_size, (255, 255, 255))
    cv2.imshow(window, img)
    cv2.waitKey(0)
