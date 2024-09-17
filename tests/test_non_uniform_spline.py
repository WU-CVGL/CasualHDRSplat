import numpy as np
import splines
import splines.quaternion
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

# load a trajectory from file
traj: PoseTrajectory3D = file_interface.read_tum_trajectory_file(
    "tests/data/traj.txt"
)

# create non-uniform splines for the position and orientation
pos_spline = splines.KochanekBartels(
    traj.positions_xyz,
    grid=traj.timestamps,
    tcb=(0, 0, 0),
    endconditions="natural"
)
rot_spline = splines.quaternion.KochanekBartels(
    [
        splines.quaternion.UnitQuaternion.from_unit_xyzw(
            np.roll(q, shift=-1)
        )
        for q in traj.orientations_quat_wxyz
    ],
    grid=traj.timestamps,
    tcb=(0, 0, 0),
    endconditions="natural"
)

# the starting and ending time of the trajectory
start_time = traj.timestamps[0]
end_time = traj.timestamps[-1]

# average frame rate of the trajectory
fps = 1 / np.mean(np.diff(traj.timestamps))

# generate a new trajectory with a uniform time grid
timestamps = np.linspace(start_time, end_time, int((end_time - start_time) * fps))
positions_xyz = pos_spline.evaluate(timestamps)
orientations_xyzw = rot_spline.evaluate(timestamps)
assert isinstance(orientations_xyzw[0], splines.quaternion.UnitQuaternion)
orientations_wxyz = np.stack(
    [np.array([q.scalar, *q.vector]) for q in orientations_xyzw],
    axis=0
)
new_traj = PoseTrajectory3D(
    timestamps=timestamps,
    positions_xyz=positions_xyz,
    orientations_quat_wxyz=orientations_wxyz
)

# save the new trajectory to file
file_interface.write_tum_trajectory_file("tests/data/traj_uniform.txt", new_traj)
