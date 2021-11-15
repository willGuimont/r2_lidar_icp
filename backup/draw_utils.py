from copy import copy

import numpy as np
from cv2 import cv2
from matplotlib import animation, pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from backup.point_cloud import PointCloud
from backup.utils import build_parallelepiped


def draw_base_vector(ax, head, text="", origin=np.array([0., 0.]), text_offset=np.array([0., 0.]), color="tab:red",
                     ha='right', va='top'):
    head_global = origin + head
    text_global = head_global + text_offset

    ax.annotate("", xy=head_global, xytext=origin,
                arrowprops=dict(arrowstyle="->,head_width=0.6, head_length=1", color=color, lw=2))
    ax.text(text_global[0], text_global[1], text, size=30, color=color, ha=ha, va=va)

    return


def draw_frame(ax, origin=np.array([0., 0.]), x=np.array([1., 0.]), y=np.array([0., 1.]), color="white",
               name="", text_x="", text_y=""):
    draw_base_vector(ax, origin=origin, head=x, text=text_x, text_offset=(0., -0.1), ha='right', va='top', color=color)
    draw_base_vector(ax, origin=origin, head=y, text=text_y, text_offset=(-0.1, 0.), ha='right', va='top', color=color)
    ax.text(origin[0], origin[1], name, color=color, size=30, ha='right', va='top')
    return


def draw_point_clouds(ax, P=None, Q=None, normals_P=None, normals_Q=None, errors=None, T=None):
    if P is not None:
        ax.scatter(P[0], P[1], alpha=0.2, color="tab:blue")
        if normals_P is not None:
            ax.quiver(P[0], P[1], normals_P[0], normals_P[1], color="yellow")
        if errors is not None:
            ax.quiver(P[0], P[1], errors[0], errors[1],
                      color="tab:red", alpha=0.4,
                      angles='xy', scale_units='xy', scale=1.)
    if Q is not None:
        ax.scatter(Q[0], Q[1], alpha=0.2, color="tab:green")
        if normals_Q is not None:
            ax.quiver(Q[0], Q[1], normals_Q[0], normals_Q[1], color="red")
    if T is not None:
        ax.quiver(0, 0, T[0, 2], T[1, 2], color="tab:red",
                  angles='xy', scale_units='xy', scale=1.)

    draw_frame(ax, x=[0.2, 0], y=[0, 0.2])
    ax.set_xlabel(r"$\vec{\mathscr{x}}$")
    ax.set_ylabel(r"$\vec{\mathscr{y}}$")
    ax.set_aspect('equal', adjustable='box')


class IcpInspector():
    def __init__(self, P=[], Q=[], T=[], I=[]):
        self.P = [copy(P)]
        self.Q = copy(Q)
        self.T = [np.copy(T)]
        self.I = [np.copy(I)]

    def append(self, P, T, I):
        self.P.append(copy(P))
        self.T.append(np.copy(T))
        self.I.append(np.copy(I))

    def draw_icp_frame(self, i):
        P_pts = self.P[i].features[:2, :]
        Q_pts = self.Q.features[:2, :]
        indices = self.I[i]
        errors = Q_pts[:, indices] - P_pts

        self.h_P.set_offsets(P_pts.T)
        self.h_err.set_offsets(P_pts.T)
        self.h_err.set_UVC(errors[0], errors[1])
        self.h_err_vec.set_UVC(errors[0], errors[1])
        self.time_text.set_text("iteration " + str(i))

        return (self.h_P,
                self.h_err,
                self.h_err_vec,
                self.time_text)

    def build_animation(self):
        nb_frames = len(self.P)

        nb_pts = len(self.P[0].features[0])
        zeros = np.zeros(nb_pts)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax = axs[0]
        ax.scatter(self.Q.features[0], self.Q.features[1], alpha=0.2, color="tab:green")
        self.h_P = ax.scatter(self.P[0].features[0], self.P[0].features[1], alpha=0.2, color="tab:blue")
        self.h_err = ax.quiver(zeros, zeros,
                               zeros, zeros,
                               color="tab:red", alpha=0.4,
                               angles='xy', scale_units='xy', scale=1.)

        ax.set_xlabel(r"$\vec{\mathscr{x}}$")
        ax.set_ylabel(r"$\vec{\mathscr{y}}$")
        ax.set_title("Point clouds")
        ax.set_aspect('equal', adjustable='box')
        draw_frame(ax, x=[0.2, 0], y=[0, 0.2])

        ax = axs[1]
        self.h_err_vec = ax.quiver(zeros, zeros,
                                   zeros, zeros,
                                   color="tab:red", alpha=0.2,
                                   angles='xy', scale_units='xy', scale=1.)
        self.time_text = ax.text(0.05, 0.05, '', fontsize=20, transform=ax.transAxes)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r"Error on $\vec{\mathscr{x}}$")
        ax.set_ylabel(r"Error on $\vec{\mathscr{y}}$")
        ax.set_title("Residual errors")
        lim = [-0.25, 0.25]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        fig.tight_layout()

        anim = animation.FuncAnimation(fig, self.draw_icp_frame,
                                       frames=np.arange(nb_frames), interval=1000,
                                       blit=True, repeat=True)
        plt.close(anim._fig)
        return anim


def draw_parallelepiped(ax, P, *args, **kwargs):
    if (P.shape[0] == 3):
        vertices = build_parallelepiped(P)
        ax.add_collection3d(Poly3DCollection(vertices, *args, **kwargs))
    else:
        print("The entered point cloud has invalid dimensions")


def draw_3d_point_clouds(ax, P, Q, errors):
    draw_parallelepiped(ax, Q[0:3], fc='white', lw=1, edgecolors='tab:green', alpha=.2)
    draw_parallelepiped(ax, P[0:3], fc='white', lw=1, ls=':', edgecolors='red', alpha=.2)

    ax.quiver(P[0], P[1], P[2], errors[0], errors[1], errors[2], color="tab:red", arrow_length_ratio=0.05)

    # cosmetics
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax_lim = 3.
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    pane_color = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    grid_color = (1.0, 1.0, 1.0, 0.2)
    ax.xaxis._axinfo["grid"]['color'] = grid_color
    ax.yaxis._axinfo["grid"]['color'] = grid_color
    ax.zaxis._axinfo["grid"]['color'] = grid_color


def draw_point_cloud_cv2(pc: PointCloud, img, size: int, color: (int, int, int), scaling_factor=10_000, offset=(0, 0)):
    assert pc.features.shape[0] == 3, "only works with 2d points"
    tmp = np.copy(pc.features)
    tmp = tmp / tmp[2, :]
    for (x, y) in tmp[:2, :].T:
        x = int(x / scaling_factor * size + size / 2) + offset[0]
        y = int(y / scaling_factor * size + size / 2) + offset[1]
        cv2.circle(img, (x, y), 3, color, -1)
