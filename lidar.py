import cv2.cv2 as cv2
import numpy as np
from rplidar import RPLidar

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

    for (x, y) in zip(xs, ys):
        x = int(x / 6500 * window_size)
        y = int(y / 6500 * window_size)

        x += window_size // 2
        y += window_size // 2

        cv2.circle(img, (x, y), 5, (255, 255, 0), -1)

    cv2.imshow(window, img)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

lidar.stop()
lidar.stop_motor()
lidar.disconnect()
