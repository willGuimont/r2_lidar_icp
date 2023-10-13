import pickle

import cv2
import numpy as np
from rplidar import RPLidar

from r2_lidar_icp.point_cloud.point_cloud import PointCloud
from r2_lidar_icp.utils.draw_utils import draw_point_cloud_cv2

if __name__ == '__main__':
    lidar = RPLidar('/dev/ttyUSB0')
    window = "lidar"
    window_size = 500

    cv2.namedWindow(window)
    for i, scan in enumerate(lidar.iter_scans()):
        img = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        pc = PointCloud.from_scan(scan)

        draw_point_cloud_cv2(pc, img, window_size, (255, 255, 0))

        cv2.imshow(window, img)
        key = cv2.waitKey(1)
        if key == ord(' '):
            pickle.dump(scan, open(f'data/live/{i}.pkl', 'wb'))
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
