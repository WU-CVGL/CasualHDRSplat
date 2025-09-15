import matplotlib.pyplot as plt
from evo.tools import file_interface
from evo.tools import plot
from evo.core.sync import associate_trajectories
import camera_plot


def test_compare_two_trajactories(traj_01_, traj_gt_, marker_scale=0.005, show_camera_indices=False):
    a = plot.PlotCollection("test")
    fig = plt.figure()
    plot_mode = plot.PlotMode["xyz"]
    ax = plot.prepare_axis(fig, plot_mode)

    plot.traj(ax, plot_mode, traj_01_,
              style="-",
              color="r",
              label="method 01",
              alpha=0.5)

    plot.traj(ax, plot_mode, traj_gt_,
              style="--",
              color="gray",
              label="GT",
              alpha=0.5)

    if show_camera_indices:
        camera_plot.draw_camera_indices(ax, traj_01_, color="r")
        camera_plot.draw_camera_indices(ax, traj_gt_, color="gray")

    # marker_scale is the size of the pose pyramids
    camera_plot.compare_two_trajactories(
        ax, traj_gt_, traj_01_, marker_scale=marker_scale)

    # This is an alternative to the pyramids
    # plot.draw_coordinate_axes(ax, traj_01_, plot_mode, marker_scale)
    # plot.draw_coordinate_axes(ax, traj_gt_, plot_mode, marker_scale)

    # disable axis grid
    plt.axis('off')

    a.add_figure("compare two trajactories", fig)
    a.show()


def test_compare_three_trajactories(traj_01_, traj_02_, traj_gt_, marker_scale=0.005, show_camera_indices=False):
    a = plot.PlotCollection("test")
    fig = plt.figure()
    plot_mode = plot.PlotMode["xyz"]
    ax = plot.prepare_axis(fig, plot_mode)

    plot.traj(ax, plot_mode, traj_01_,
              style="-",
              color="r",
              label="method 01",
              alpha=0.5)

    plot.traj(ax, plot_mode, traj_02_,
              style="-",
              color="b",
              label="method 2",
              alpha=0.5)

    plot.traj(ax, plot_mode, traj_gt_,
              style="--",
              color="gray",
              label="GT",
              alpha=0.5)

    if show_camera_indices:
        camera_plot.draw_camera_indices(ax, traj_01_, color="r")
        camera_plot.draw_camera_indices(ax, traj_02_, color="b")
        camera_plot.draw_camera_indices(ax, traj_gt_, color="gray")

    # This is an alternative to the pyramids
    # plot.draw_coordinate_axes(ax, traj_01_, plot_mode, marker_scale)
    # plot.draw_coordinate_axes(ax, traj_02_, plot_mode, marker_scale)
    # plot.draw_coordinate_axes(ax, traj_gt_, plot_mode, marker_scale)

    # marker_scale is the size of the pose pyramids
    camera_plot.draw_pyramids(ax, traj_01_, marker_scale=marker_scale, color='r')
    camera_plot.draw_pyramids(ax, traj_02_, marker_scale=marker_scale, color='b')
    camera_plot.draw_pyramids(ax, traj_gt_, marker_scale=marker_scale, color='gray')

    # disable axis grid
    plt.axis('off')

    a.add_figure("compare three trajactories", fig)
    a.show()


def main():
    FILE_GT = "/home/cvgluser/blender/blender-3.6.13-linux-x64/data/deblurnerf/rawdata_new_tra1/tanabata/trajectories_path/24.txt"
    FILE_01 = "/home/cvgluser/blender/blender-3.6.13-linux-x64/data/deblurnerf/rawdata_new_tra1/tanabata/trajectories_path/24.txt"
    FILE_02 = "/home/cvgluser/blender/blender-3.6.13-linux-x64/data/deblurnerf/rawdata_new_tra1/tanabata/trajectories_path/24.txt"

    traj_gt = file_interface.read_tum_trajectory_file(FILE_GT)
    traj_01 = file_interface.read_tum_trajectory_file(FILE_01)
    traj_02 = file_interface.read_tum_trajectory_file(FILE_02)

    gt_synced01, traj_01_synced = associate_trajectories(
        traj_gt, traj_01, max_diff=0.1)
    gt_synced02, traj_02_synced = associate_trajectories(
        traj_gt, traj_02, max_diff=0.1)

    traj_01_synced.align(gt_synced01, correct_scale=True)
    traj_02_synced.align(gt_synced02, correct_scale=True)

    test_compare_two_trajactories(traj_01_synced, gt_synced01, marker_scale=0.001, show_camera_indices=False)
    test_compare_three_trajactories(traj_01_synced, traj_02_synced, traj_gt, marker_scale=0.001, show_camera_indices=False)


if __name__ == "__main__":
    main()
