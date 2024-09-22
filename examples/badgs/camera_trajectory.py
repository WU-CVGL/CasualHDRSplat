"""
Trainable camera trajectory based on SE(3) B-spline.

Created by lzzhao on 2024.04.30
"""
from __future__ import annotations

from math import ceil
from typing import Literal, Tuple, Optional, Type, Union

import pypose as pp
import splines
import splines.quaternion
import torch
from dataclasses import dataclass, field
from jaxtyping import Float, Int
from pypose import LieTensor
from torch import nn, Tensor
from typing_extensions import assert_never

from badgs.base_config import InstantiateConfig
from badgs.exposure_time_optimizer import ExposureTimeOptimizerConfig
from badgs.spline import Spline, SplineConfig

TrajSamplingMode = Literal["uniform", "mid"]


@dataclass
class CameraTrajectoryConfig(InstantiateConfig):
    """Configuration for camera trajectory instantiation."""

    _target: Type = field(default_factory=lambda: CameraTrajectory)
    """Target class to instantiate."""

    spline: SplineConfig = field(default_factory=SplineConfig)
    """Configuration for the spline trajectory."""

    exposure_time_optimizer: ExposureTimeOptimizerConfig = field(
        default_factory=lambda: ExposureTimeOptimizerConfig(mode="off")
    )
    """Configuration for the exposure time optimizer."""

    num_virtual_views: int = 10
    """The number of samples used to model the motion-blurring."""

    traj_interpolate_ratio: float = 2.0
    """The ratio of the trajectory sampling interval to the frame interval."""


class CameraTrajectory(nn.Module):
    """Trainable camera trajectory based on SE(3) B-spline.

    Args:
        config: the CameraTrajectoryConfig used to instantiate class
        timestamps: the timestamps of the corresponding captured images
        exposure_times: the exposure times of the captured images
        c2ws: the camera poses of the captured images (camera to world)
        device: cpu or cuda
    """

    config: CameraTrajectoryConfig
    spline: Spline
    timestamps: Tensor
    frame_interval: float

    def __init__(
            self,
            config: CameraTrajectoryConfig,
            timestamps: Tensor,
            exposure_times: Float[Tensor, "num_cameras"],
            c2ws: Float[Tensor, "num_cameras 3 4"] | Float[Tensor, "num_cameras 4 4"],
            device: Union[torch.device, str],
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.timestamps = timestamps.to(device)

        # First initialize a temporary spline to interpolate on, and assume it non-uniform
        self.start_time = timestamps[0].item()
        self.end_time = timestamps[-1].item()
        self.frame_interval: float = torch.mean(torch.diff(timestamps)).item()
        self.fps = 1.0 / self.frame_interval
        pos_spline = splines.KochanekBartels(
            c2ws[:, :3, 3].cpu().numpy(),
            grid=timestamps.cpu().numpy(),
            tcb=(0, 0, 0),
            endconditions="natural"
        )
        rot_spline = splines.quaternion.KochanekBartels(
            [
                splines.quaternion.UnitQuaternion.from_unit_xyzw(
                    pp.mat2SO3(rot).to("cpu").numpy()
                )
                for rot in c2ws[:, :3, :3].cpu().numpy()
            ],
            grid=timestamps.cpu().numpy(),
            tcb=(0, 0, 0),
            endconditions="natural"
        )

        # Our new optimizable cubic B-spline
        self.config.spline.start_time = self.start_time
        spline_timestamps = torch.linspace(  # TODO: implement extrapolate
            self.start_time,
            self.end_time,
            int(ceil(len(c2ws) * self.config.traj_interpolate_ratio))
        )
        spline_timestamps_numpy = spline_timestamps.cpu().numpy()
        self.config.spline.sampling_interval = self.frame_interval / self.config.traj_interpolate_ratio
        self.spline = self.config.spline.setup(
            max_knots=len(spline_timestamps),
            device=device,
        )
        self.virtual_view_multipliers = torch.linspace(
            0, 1, self.config.num_virtual_views, device="cpu"
        )

        # Initialize our spline from the temporary spline, with traj_interpolate_ratio
        positions_xyz = torch.tensor(pos_spline.evaluate(spline_timestamps_numpy))
        orientations_xyzw = rot_spline.evaluate(spline_timestamps_numpy)
        assert isinstance(orientations_xyzw[0], splines.quaternion.UnitQuaternion)
        orientations_xyzw = torch.stack(
            [torch.tensor([*q.vector, q.scalar]) for q in orientations_xyzw],
            axis=0
        )
        self.poses = pp.SE3(
            torch.cat([positions_xyz, orientations_xyzw], dim=-1)
        ).float().to(device)
        self.spline.set_data(self.poses)

        # Exposure times
        self.exposure_times = exposure_times.to(self.device)
        self.exposure_time_optimizer = self.config.exposure_time_optimizer.setup(
            num_cameras=len(exposure_times),
            device=device,
        )

    def forward(
            self,
            colmap_image_id: Float[Tensor, "*batch_size"],
            mode: TrajSamplingMode,
    ) -> Tuple[Float[LieTensor, "*batch_size 7"], Optional[float]]:
        """Get camera pose and corresponding exposure_times from the camera trajectory.

        Args:
            colmap_image_id: the index of the image in the colmap data
            mode: the mode of trajectory sampling

        Returns:
            pose(s): the camera pose (camera to world)
            exposure_time: the exposure time
        """
        timestamp = self.timestamps[colmap_image_id]
        exposure_time = self.get_exposure_time(colmap_image_id)

        if mode == "start":
            pose = self.spline(timestamp)
            return pose, exposure_time

        elif mode == "end":
            assert exposure_time is not None
            pose = self.spline(timestamp + exposure_time)
            return pose, exposure_time

        elif mode == "uniform":
            poses = self.spline(
                timestamp.to(exposure_time.device) + self.virtual_view_multipliers.to(
                    exposure_time.device) * exposure_time
            )
            return poses, exposure_time
        elif mode == "mid":
            poses = self.spline(
                timestamp.to(exposure_time.device) + 0.5 * exposure_time
            )
            return poses, exposure_time
        else:
            assert_never(mode)

    def get_exposure_time(self, colmap_image_id: Int) -> float:
        """Get exposure time from the camera trajectory.

        Args:
            colmap_image_id: the index of the image

        Returns:
            exposure_time: the exposure time
        """
        exposure_time_opt = self.exposure_time_optimizer(colmap_image_id)
        exposure_time = self.exposure_times[colmap_image_id] + exposure_time_opt
        return torch.clamp(exposure_time,min=1e-5, max=self.frame_interval)

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        self.spline.get_loss_dict(loss_dict)
        self.exposure_time_optimizer.get_loss_dict(loss_dict)

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        self.spline.get_metrics_dict(metrics_dict)
        self.exposure_time_optimizer.get_metrics_dict(metrics_dict)

    def get_param_groups(self, param_groups: dict) -> None:
        """Get spline parameters"""
        self.spline.get_param_groups(param_groups)
        self.exposure_time_optimizer.get_param_groups(param_groups)
