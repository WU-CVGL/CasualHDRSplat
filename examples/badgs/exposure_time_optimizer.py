"""
Exposure time optimizer.
"""

from __future__ import annotations

from typing import Literal, Type, Union

import torch
from dataclasses import dataclass, field
from jaxtyping import Int
from torch import nn, Tensor
from typing_extensions import assert_never

from badgs.base_config import InstantiateConfig


@dataclass
class ExposureTimeOptimizerConfig(InstantiateConfig):
    """Configuration for exposure time optimizer instantiation."""

    _target: Type = field(default_factory=lambda: ExposureTimeOptimizer)
    """Target class to instantiate."""

    mode: Literal["off", "linear"] = "off"
    """Exposure time optimization strategy to use."""

    initial_noise_std: float = 1e-5
    """Initial perturbation to exposure time delta."""


class ExposureTimeOptimizer(nn.Module):
    """Exposure time optimizer.

    Args:
        config: the ExposureTimeOptimizerConfig used to instantiate class
        num_cameras: the number of cameras
        device: cpu or cuda
    """

    config: ExposureTimeOptimizerConfig

    def __init__(
            self,
            config: ExposureTimeOptimizerConfig,
            num_cameras: Int,
            device: Union[torch.device, str],
    ):
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device

        # Initialize learnable parameters.
        if self.config.mode == "off":
            return
        elif self.config.mode == "linear":
            self.adjustment = nn.Parameter(
                torch.randn(self.num_cameras, device=self.device) * self.config.initial_noise_std
            )
        else:
            assert_never(self.config.mode)

    def forward(self, indices: Int[Tensor, "camera_indices"]):
        """Get the exposure time adjustment for the given camera indices.

        Args:
            indices: The indices of the cameras.

        Returns:
            The exposure time adjustment.
        """
        if self.config.mode == "off":
            return torch.zeros_like(indices, device=self.device)
        elif self.config.mode == "linear":
            return self.adjustment[indices]
        else:
            assert_never(self.config.mode)

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        pass

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            metrics_dict["exposure_time_opt"] = torch.mean(self.adjustment).item()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get exposure time optimizer parameters"""
        spline_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(spline_opt_params) > 0
            param_groups["exposure_time_opt"] = spline_opt_params
        else:
            assert len(spline_opt_params) == 0
