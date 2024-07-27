"""
Trainable camera trajectory based on SE(3) B-spline.

Created by lzzhao on 2024.04.30
"""
from __future__ import annotations

from copy import deepcopy
from math import ceil
from typing import Literal, Tuple, Optional, Type, Union

import pypose as pp
import torch
from badgs.spline import Spline, SplineConfig
from dataclasses import dataclass, field
from jaxtyping import Float, Int
from nerfstudio.configs.base_config import InstantiateConfig
from pypose import LieTensor
from torch import nn, Tensor
from typing_extensions import assert_never

TrajSamplingMode = Literal["uniform", "mid"]


@dataclass
class CameraTrajectoryConfig(InstantiateConfig):
    """Configuration for camera trajectory instantiation."""

    _target: Type = field(default_factory=lambda: CameraTrajectory)
    """Target class to instantiate."""

    spline: SplineConfig = field(default_factory=SplineConfig)
    """Configuration for the spline trajectory."""

    optimize_exposure_time: bool = False
    """Whether to enable exposure time as a parameter."""

    num_virtual_views: int = 10
    """The number of samples used to model the motion-blurring."""

    traj_interpolate_ratio: float = 2.0
    """The ratio of the trajectory sampling interval to the frame interval."""


class CameraTrajectory(nn.Module):
    """Trainable camera trajectory based on SE(3) B-spline.

    Args:
        config: the CameraTrajectoryConfig used to instantiate class
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
        self.timestamps = timestamps

        # Spline
        self.config.spline.start_time = timestamps[0].item()
        self.frame_interval = (timestamps[1] - timestamps[0]).item()

        # temporary spline to interpolate for the camera trajectory initialization
        temp_spline_config = deepcopy(self.config.spline)
        temp_spline_config.sampling_interval = self.frame_interval
        temp_spline_config.spline_optimizer.mode = "off"
        temp_spline = temp_spline_config.setup(
            max_knots=len(c2ws),
            device=device,
        )
        temp_spline.set_data(pp.mat2SE3(c2ws).to(device))

        # Spline
        spline_timestamps = torch.linspace(  # TODO: implement extrapolate
            timestamps[1].item(),
            timestamps[-2].item(),
            int(ceil(len(c2ws) * self.config.traj_interpolate_ratio))
        )  # type: ignore
        self.config.spline.sampling_interval = self.frame_interval / self.config.traj_interpolate_ratio
        self.spline = self.config.spline.setup(
            max_knots=len(spline_timestamps),
            device=device,
        )
        self.virtual_view_multipliers = torch.linspace(
            0, 1, self.config.num_virtual_views, device="cpu"
        )

        # Spline poses
        self.poses = temp_spline(spline_timestamps).to(device)
        self.spline.set_data(self.poses)

        # Exposure times
        if self.config.optimize_exposure_time:
            self.exposure_times = nn.Parameter(exposure_times)
        else:
            self.exposure_times = exposure_times

    def forward(
            self,
            image_idx: Float[Tensor, "*batch_size"],
            mode: TrajSamplingMode,
    ) -> Tuple[Float[LieTensor, "*batch_size 7"], Optional[float]]:
        """Get camera pose and corresponding exposure_times from the camera trajectory.

        Args:
            image_idx: the index of the image
            mode: the mode of trajectory sampling

        Returns:
            pose(s): the camera pose (camera to world)
            exposure_time: the exposure time
        """
        timestamp = self.timestamps[image_idx]
        exposure_time = self.get_exposure_time(image_idx)

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

    def get_exposure_time(self, image_idx: Int) -> float:
        """Get exposure time from the camera trajectory.

        Args:
            image_idx: the index of the image

        Returns:
            exposure_time: the exposure time
        """
        exposure_time = self.exposure_times[image_idx]
        return torch.clamp(exposure_time, min=1e-5, max=self.frame_interval)

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        pass

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        self.spline.get_metrics_dict(metrics_dict)
        metrics_dict["exposure_times"] = torch.mean(self.exposure_times).item()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get spline parameters"""
        self.spline.get_param_groups(param_groups)
        if self.config.optimize_exposure_time:
            param_groups["exposure_times"] = [self.exposure_times]
