import cv2.cv2 as cv2
import numpy as np

from backup.point_cloud.diff_icp import point_to_homogeneous
from backup.point_cloud import TransformationMatrix


def line_line_intersection(p1, p2, p3, p4):
    denominator = (p4[0] - p3[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p4[1] - p3[1])
    if denominator == 0:
        return None
    t1 = ((p3[1] - p4[1]) * (p1[0] - p3[0]) + (p4[0] - p3[0]) * (p1[1] - p3[1])) / denominator
    t2 = ((p1[1] - p2[1]) * (p1[0] - p3[0]) + (p2[0] - p1[0]) * (p1[1] - p3[1])) / denominator
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return p1 + t1 * (p2 - p1)
    else:
        return None


# settings
arena_size = 10.
arena_inner_size = 8.
obstacle_size = 2
walls = np.array([
    # outer edge
    [[-arena_size, -arena_inner_size], [-arena_inner_size, -arena_size]],
    [[-arena_inner_size, -arena_size], [arena_inner_size, -arena_size]],
    [[arena_inner_size, -arena_size], [arena_size, -arena_inner_size]],
    [[arena_size, -arena_inner_size], [arena_size, arena_inner_size]],
    [[arena_size, arena_inner_size], [arena_inner_size, arena_size]],
    [[arena_inner_size, arena_size], [-arena_inner_size, arena_size]],
    [[-arena_inner_size, arena_size], [-arena_size, arena_inner_size]],
    [[-arena_size, arena_inner_size], [-arena_size, -arena_inner_size]],

    # inner obstacle
    [[-obstacle_size, -obstacle_size], [obstacle_size, -obstacle_size]],
    [[obstacle_size, -obstacle_size], [obstacle_size, obstacle_size]],
    [[obstacle_size, obstacle_size], [-obstacle_size, obstacle_size]],
    [[-obstacle_size, obstacle_size], [-obstacle_size, -obstacle_size]],
])
robot_pos = np.array([0., -5])
robot_yaw = 0
lidar_range = 15
num_beam = 32

# gui stuff
window = "lidar"
window_size = 500
cv2.namedWindow(window)
scale = 10
world_to_win = TransformationMatrix() \
    .translate(window_size / 2, window_size / 2) \
    .scale(scale) \
    .build()
win_to_world = np.linalg.inv(world_to_win)
visual_lidar_range = lidar_range * scale
robot_radius = 10


def line_to_window(line):
    return (world_to_win @ point_to_homogeneous(line))[:2].astype(int)


def point_to_window(pt):
    return (world_to_win @ point_to_homogeneous(np.array([pt])))[:2].astype(int)[:, 0]


if __name__ == '__main__':
    img = np.zeros((window_size, window_size, 3), dtype=np.uint8)
    while True:
        img[:, :, :] = 0

        # draw walls
        for wall in walls:
            wall_start, wall_end = line_to_window(wall).T
            cv2.line(img, wall_start, wall_end, color=(255, 255, 0))

        # lidar
        robot_pos_window = point_to_window(robot_pos)
        intersections = []
        for i in range(num_beam):
            theta = robot_yaw + i * 2 * np.pi / num_beam
            lidar_end_x, lidar_end_y = np.cos(theta) * lidar_range + robot_pos[0] + robot_pos[0], \
                                       np.sin(theta) * lidar_range + robot_pos[1] + robot_pos[1]
            line = robot_pos, np.array([lidar_end_x, lidar_end_y])
            best_inter = None
            smallest_distance = np.inf
            for wall in walls:
                wall_start, wall_end = wall
                inter = line_line_intersection(*line, wall_start, wall_end)
                if inter is not None:
                    distance = np.linalg.norm(np.array(inter) - robot_pos)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        best_inter = inter

            if best_inter is not None:
                intersections.append(np.copy(best_inter))

            if i % 1 == 0:
                if best_inter is None:
                    rx, ry = point_to_window(line[1])
                else:
                    rx, ry = point_to_window(best_inter)
                cv2.line(img, (robot_pos_window[0], robot_pos_window[1]), (rx, ry), (0, 255, 0), 1)

        # robot
        cv2.circle(img, (robot_pos_window[0], robot_pos_window[1]), robot_radius, (255, 0, 255), -1)
        robot_front_x, robot_front_y = int(np.cos(robot_yaw) * robot_radius + robot_pos_window[0]), \
                                       int(np.sin(robot_yaw) * robot_radius + robot_pos_window[1])
        cv2.line(img, (robot_pos_window[0], robot_pos_window[1]), (robot_front_x, robot_front_y), (255, 0, 0), 2)

        # points
        for inter in intersections:
            inter = point_to_window(inter)
            cv2.circle(img, (inter[0], inter[1]), 5, (255, 0, 127), -1)

        cv2.imshow(window, img)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
