import pickle

import cv2.cv2 as cv2
import numpy as np
from rplidar import RPLidar

from tools.visualize_point_cloud import draw_point_cloud

if __name__ == '__main__':
    lidar = RPLidar('/dev/ttyUSB0')
    window = "lidar"
    window_size = 500

    cv2.namedWindow(window)
    for i, scan in enumerate(lidar.iter_scans()):
        img = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        scan = np.array(scan)

        qualities, angles, distances = scan[:, 0], scan[:, 1], scan[:, 2]
        angles = np.deg2rad(angles)

        xs = np.cos(angles) * distances
        ys = np.sin(angles) * distances

        pc = np.stack((xs, ys))
        draw_point_cloud(pc, img, window_size, (255, 255, 0))

        cv2.imshow(window, img)
        key = cv2.waitKey(1)
        if key == ord(' '):
            pickle.dump(scan, open(f'scans/{i}.pkl', 'wb'))
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
