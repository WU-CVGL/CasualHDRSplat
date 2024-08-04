"""
SE(3) B-Spline optimizer.

Created by lzzhao on 2024.04.30
"""

from __future__ import annotations

from typing import Literal, Type, Union

import pypose as pp
import torch
from dataclasses import dataclass, field
from jaxtyping import Float, Int
from pypose import LieTensor
from torch import nn, Tensor
from typing_extensions import assert_never

from badgs.base_config import InstantiateConfig


@dataclass
class SplineOptimizerConfig(InstantiateConfig):
    """Configuration for spline optimizer instantiation."""

    _target: Type = field(default_factory=lambda: SplineOptimizer)
    """Target class to instantiate."""

    mode: Literal["off", "SE3"] = "off"
    """Pose optimization strategy to use."""

    initial_noise_se3_std: float = 1e-5
    """Initial perturbation to pose delta on se(3). Must be non-zero to prevent NaNs."""


class SplineOptimizer(nn.Module):
    """SE(3) B-Spline optimizer.

    Args:
        config: the SplineOptimizerConfig used to instantiate class
        num_cameras: the number of cameras
        device: cpu or cuda
    """

    config: SplineOptimizerConfig

    def __init__(
            self,
            config: SplineOptimizerConfig,
            num_cameras: Int,
            device: Union[torch.device, str],
    ):
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.dof = 6
        """Degrees of freedom of manifold, i.e. number of dimensions of the tangent space"""
        self.dim = 7
        """Dimentions of pose parameterization. Three for translation, 4-tuple for quaternion"""

        # Initialize learnable parameters.
        if self.config.mode == "off":
            return
        elif self.config.mode == "SE3":
            self.pose_adjustment = pp.Parameter(
                pp.randn_se3(
                    num_cameras,
                    sigma=self.config.initial_noise_se3_std,
                    device=device,
                ),
            )
        else:
            assert_never(self.config.mode)

    def forward(self, indices: Int[Tensor, "camera_indices"]) -> Float[LieTensor, "num_cameras 7"]:
        """Forward pass of the optimizer.

        Args:
            indices: The indices of the cameras to optimize.

        Returns:
            The optimized camera poses.
        """
        ret = None
        if self.config.mode == "off":
            pass
        else:
            indices = indices.int()
            unique_indices, lut = torch.unique(indices, return_inverse=True)
            camera_opt = self.pose_adjustment[unique_indices].Exp()
            ret = camera_opt[lut]
        if ret is None:
            ret = pp.identity_SE3(
                *(indices.shape[0], self.num_control_knots),
                device=self.pose_adjustment.device
            )
        return ret

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        pass

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Add spline optimizer metrics"""
        metrics_dict["camera_opt_translation"] = self.pose_adjustment[:, :3].norm()
        metrics_dict["camera_opt_rotation"] = self.pose_adjustment[:, 3:].norm()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        spline_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(spline_opt_params) > 0
            param_groups["camera_opt"] = spline_opt_params
        else:
            assert len(spline_opt_params) == 0
