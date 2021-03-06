from typing import Optional, Callable

import numpy as np


def l2_norm(x: np.ndarray, y: np.ndarray):
    return ((x - y) ** 2).sum(axis=0)


def furthest_point_sampling(pts: np.ndarray,
                            k: int,
                            initial_idx: Optional[int] = None,
                            metric: Callable[[np.ndarray, np.ndarray], float] = l2_norm,
                            skip_initial: bool = False) -> (np.ndarray, np.ndarray):
    """
    Furthest point sampling algorithm
    :param pts: np.ndarray of dimension DxN, where N is the number of points and D the dimension each points
    :param k: number of points to sample
    :param initial_idx: index to start the sampling from, random if None
    :param metric: metric function to calculate distance
    :param skip_initial: skip the first furthest point, stabilizes the output
    :return: (indices, distances), indices = sampled points
    """
    dim, num_points = pts.shape
    indices = np.zeros((k,), dtype=int)
    distances = np.zeros((k, num_points), dtype=pts.dtype)
    if initial_idx is None:
        indices[0] = np.random.randint(len(pts))
    else:
        indices[0] = initial_idx

    furthest_point = pts[:, indices[0]].T
    min_distances = metric(furthest_point[:, None], pts)
    if skip_initial:
        indices[0] = np.argmax(min_distances, axis=0)
        furthest_point = pts[:, indices[0]]
        min_distances = metric(furthest_point[:, None], pts)

    distances[0, :] = min_distances
    for i in range(1, k):
        indices[i] = np.argmax(min_distances)
        furthest_point = pts[:, indices[i]]
        dist = metric(furthest_point[:, None], pts)
        distances[:] = dist
        min_distances = np.minimum(min_distances, dist)
    return indices, distances


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    k = 30
    t = np.repeat(np.linspace(0, 2 * np.pi, num=100), 3)
    x = np.cos(t) * 5
    y = np.sin(t) * 2
    pts = np.stack((x, y), axis=1) + np.random.random((x.shape[0], 2))
    pts = pts.T
    print(pts.shape)
    indices, distances = furthest_point_sampling(pts, k, skip_initial=True)
    reduced = pts[:, indices]

    print(f'Reduced {x.shape[0]} points with {k} points')

    plt.scatter(pts[0, :], pts[1, :], c='r')
    plt.scatter(reduced[0, :], reduced[1, :], c='b', marker='x')
    plt.show()
