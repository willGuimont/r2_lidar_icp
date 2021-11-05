import cv2.cv2 as cv2
import numpy as np


def draw_point_cloud(pc, img, size: int, color: (int, int, int)):
    tmp = np.copy(pc)
    if tmp.shape[0] == 4:
        tmp = tmp / tmp[3, :]
    for (x, y) in tmp[:2].T:
        # x = int(x / 6500 * size + size / 2)
        # y = int(y / 6500 * size + size / 2)
        x = int(x / 10_000 * size + size / 2)
        y = int(y / 10_000 * size + size / 2)

        cv2.circle(img, (x, y), 3, color, -1)
