import itertools
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from evo.core import trajectory


# basic operations of transforming 3D points between world/camera/image coordinates
def to_hom(X):
    X_hom = np.concatenate([X, np.ones_like(X[..., :1])], axis=-1)
    return X_hom


def world2cam(X, pose):  # [B,N,3]
    X_hom = to_hom(X)
    pose_inv = invert(pose)
    return X_hom @ pose_inv.transpose(-1, -2)


def cam2img(X, cam_intr):
    return X @ cam_intr.transpose(-1, -2)


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = invert(pose)
    return X_hom @ pose.transpose(-1, -2)


def invert(pose, use_inverse=False):
    R, t = pose[..., :3], pose[..., 3:]
    R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
    t_inv = (-R_inv @ t)[..., 0]
    pose_inv = np.concatenate([R_inv, t_inv[..., None]], axis=-1)  # [...,3,4]
    return pose_inv


def get_camera_mesh(pose, scale=1.):
    if pose.shape[-2:] == (4, 4):
        pose = pose[:3, :]
    assert (pose.shape[-2:] == (3, 4))
    vertices = np.array([[-0.5, -0.5, 1],
                         [0.5, -0.5, 1],
                         [0.5, 0.5, 1],
                         [-0.5, 0.5, 1],
                         [0, 0, 0]]) * scale
    vertices = cam2world(vertices[None], pose)
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return wireframe


def draw_camera_indices(ax, traj, fontsize=10, color="k"):
    """Draw camera indices into the pyplot"""
    for i, pose in enumerate(traj.poses_se3):
        x, y, z = pose[:3, 3]
        ax.text(x, y, z, str(i), fontsize=fontsize, color=color)


def draw_pyramids(ax: plt.Axes,
                  traj: trajectory.PosePath3D,
                  marker_scale: float = 1.,
                  color: str = "r") -> None:
    """
    Draws a pyramid for each pose of a trajectory.\n
    :param ax: plot axis
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param marker_scale: affects the size of the marker (1. * marker_scale)
    :param color: color of the pyramid
    """
    if marker_scale <= 0:
        return

    combinations = lambda l: [[l[i], l[i + 1]] for i in range(len(l) - 1)]

    segs_3d = list(itertools.chain.from_iterable([combinations(
        [tuple(c) for c in np.array(
            get_camera_mesh(p, marker_scale)).reshape((-1, 3))])
        for p in traj.poses_se3]))

    colors = [color] * len(segs_3d)
    line_collection = art3d.Line3DCollection(segs_3d, colors=colors,
                                             alpha=1.,
                                             linestyles="solid")
    ax.add_collection(line_collection)


def plot_poses_blender(ax: Axes3D,
                       pose: np.array,
                       pose_ref: np.array,
                       marker_scale: float = 1.) -> None:
    # get the camera meshes
    cam = get_camera_mesh(pose, marker_scale)
    cam_ref = get_camera_mesh(pose_ref, marker_scale)

    # plot the cameras
    N = len(cam)
    ref_color = (0.7, 0.2, 0.7)
    pred_color = (0, 0.6, 0.7)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref], alpha=0.2, facecolor=ref_color))
    for i in range(N):
        ax.plot(cam_ref[i, :, 0], cam_ref[i, :, 1], cam_ref[i, :, 2], color=ref_color, linewidth=0.5)
        ax.scatter(cam_ref[i, 5, 0], cam_ref[i, 5, 1], cam_ref[i, 5, 2], color=ref_color, s=20)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam], alpha=0.2, facecolor=pred_color))
    for i in range(N):
        ax.plot(cam[i, :, 0], cam[i, :, 1], cam[i, :, 2], color=pred_color, linewidth=1)
        ax.scatter(cam[i, 5, 0], cam[i, 5, 1], cam[i, 5, 2], color=pred_color, s=20)
    for i in range(N):
        ax.plot([cam[i, 5, 0], cam_ref[i, 5, 0]],
                [cam[i, 5, 1], cam_ref[i, 5, 1]],
                [cam[i, 5, 2], cam_ref[i, 5, 2]], color=(1, 0, 0), linewidth=3)


def compare_two_trajactories(ax: Axes3D,
                             traj: trajectory.PosePath3D,
                             traj_ref: trajectory.PosePath3D,
                             marker_scale: float = 1.) -> None:
    """
    Compare two trajectories with little pyramid meshes.\n
    :param ax: 3D plot axis
    :param traj: will be colored in blue
    :param traj_ref: will be colored in red
    :param scale: scale factor of pyramids
    """
    assert(traj.num_poses == traj_ref.num_poses)
    for p1, p2 in zip(traj.poses_se3, traj_ref.poses_se3):
        plot_poses_blender(ax, p1, p2, marker_scale)
