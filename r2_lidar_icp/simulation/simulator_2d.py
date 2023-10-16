from dataclasses import dataclass
from typing import Optional

import numpy as np

from r2_lidar_icp.utils.utils import point_to_homogeneous, line_line_intersection_2d


@dataclass
class World2D:
    walls: np.ndarray


@dataclass
class Robot:
    robot_pos: np.ndarray
    robot_yaw: float
    lidar_range: float
    num_beam: int


@dataclass
class Beam:
    angle: float
    beam_start: np.ndarray
    beam_end: np.ndarray
    hit: Optional[np.ndarray]


class Simulator2D:
    def __init__(self, world: World2D):
        """
        Create a 2D simulator.
        :param world: 2D world
        """
        self.world = world

    def cast(self, robot: Robot) -> [Beam]:
        """
        Cast a lidar beam.
        :param robot: Robot
        :return: List of beams
        """
        beams = []
        for i in range(robot.num_beam):
            theta = robot.robot_yaw + i * 2 * np.pi / robot.num_beam
            beam = self.cast_beam(robot, theta)
            beams.append(beam)
        return beams

    def cast_beam(self, robot: Robot, theta: float) -> Beam:
        lidar_end_x, lidar_end_y = np.cos(theta) * robot.lidar_range + robot.robot_pos[0], \
                                   np.sin(theta) * robot.lidar_range + robot.robot_pos[1]
        beam_start = robot.robot_pos
        beam_end = np.array([lidar_end_x, lidar_end_y])

        best_inter = None
        smallest_distance = np.inf
        for wall in world.walls:
            wall_start, wall_end = wall
            inter = line_line_intersection_2d(beam_start, beam_end, wall_start, wall_end)
            if inter is not None:
                distance = np.linalg.norm(np.array(inter) - beam_start)
                if distance < smallest_distance:
                    smallest_distance = distance
                    best_inter = inter
        return Beam(theta, beam_start, beam_end, best_inter)


if __name__ == '__main__':
    import cv2
    from r2_lidar_icp.utils.transformation_matrix import TransformationMatrix2D

    # World
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
    world = World2D(walls)

    # Robot
    robot_pos = np.array([[5, 0]]).T
    robot_yaw = 0
    lidar_range = 10
    num_beam = 32
    robot = Robot(robot_pos, robot_yaw, lidar_range, num_beam)

    sim = Simulator2D(world)

    # GUI
    window = 'lidar'
    window_size = 500
    cv2.namedWindow(window)
    scale = 10
    world_to_win = TransformationMatrix2D() \
        .translate(window_size / 2, window_size / 2) \
        .scale(scale) \
        .build()
    win_to_world = np.linalg.inv(world_to_win)
    robot_radius = 10


    def line_to_window(line):
        return (world_to_win @ point_to_homogeneous(line.T))[:2].astype(int)


    def point_to_window(pt):
        return (world_to_win @ point_to_homogeneous(pt))[:2].astype(int)[:, 0]


    time = 0
    while True:
        time += 1
        robot.robot_pos = np.array([[np.cos(time / 100) * 6, np.sin(time / 100) * 6]]).T
        robot_yaw = time / 100 + np.pi / 2

        img = np.zeros((window_size, window_size, 3), dtype=np.uint8)

        # draw walls
        for wall in walls:
            wall_start, wall_end = line_to_window(wall).T
            cv2.line(img, wall_start, wall_end, color=(255, 255, 0))

        # lidar
        beams = sim.cast(robot)
        for b in beams:
            (sx, sy) = point_to_window(b.beam_start)
            (ex, ey) = point_to_window(b.hit if b.hit is not None else b.beam_end)
            cv2.line(img, (sx, sy), (ex, ey), (0, 255, 255), 1)
            if b.hit is not None:
                cv2.circle(img, (ex, ey), 5, (0, 255, 255), -1)

        # robot
        robot_pos_window = point_to_window(robot.robot_pos)
        cv2.circle(img, (robot_pos_window[0], robot_pos_window[1]), robot_radius, (255, 0, 255), -1)
        robot_front_x, robot_front_y = int(np.cos(robot_yaw) * robot_radius + robot_pos_window[0]), \
            int(np.sin(robot_yaw) * robot_radius + robot_pos_window[1])
        cv2.line(img, (robot_pos_window[0], robot_pos_window[1]), (robot_front_x, robot_front_y), (255, 0, 0), 2)

        cv2.imshow(window, img)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
