import matplotlib.pyplot as plt
import numpy as np
from rplidar import RPLidar

lidar = RPLidar('/dev/ttyUSB0')

info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)


def plot_scan(scan):
    quality, angle, distance = scan[:, 0], scan[:, 1], scan[:, 2]
    angle = np.deg2rad(angle)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(angle, distance)
    ax.set_rlabel_position(-22.5)
    ax.grid(True)

    ax.set_title('LiDAR')
    plt.show()


for i, scan in enumerate(lidar.iter_scans()):
    scan = np.array(scan)
    plot_scan(scan)
    if i > 10:
        break

lidar.stop()
lidar.stop_motor()
lidar.disconnect()
