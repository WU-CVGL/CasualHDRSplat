import random

import pypose as pp
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F

from spline_function import bezier_interpolation

class PoseOptModule(torch.nn.Module):
    """Camera pose optimization module w/ Bezier curve"""

    def __init__(self, camera_to_worlds, traj_mode='bezier', bezier_degree=10, initial_noise=1e-5, pose_refine=False, pose_noise=0):
        super().__init__()
        self.pose_refine = pose_refine
        self.num_views = camera_to_worlds.shape[0]
        self.traj_mode = traj_mode
        
        if not pose_refine:
            if traj_mode == 'bezier':
                poses_control_knots_SE3 = pp.mat2SE3(camera_to_worlds, check=False) # NOTE: hack for now
                poses_control_knots_SE3_mid = poses_control_knots_SE3[self.num_views//2]
                poses_control_knots_SE3 = poses_control_knots_SE3_mid.unsqueeze(0).repeat(bezier_degree,1).unsqueeze(0)
                poses_control_knots_se3 = poses_control_knots_SE3.Log()

                poses_noise_se3 = pp.randn_se3(1, bezier_degree, sigma=initial_noise)
                poses_control_knots_se3 += poses_noise_se3

                self.pose_control_knots = pp.Parameter(poses_control_knots_se3)
            elif traj_mode == 'individual':
                poses_control_knots_SE3 = pp.mat2SE3(camera_to_worlds, check=False) # NOTE: hack for now
                poses_control_knots_SE3_mid = poses_control_knots_SE3[self.num_views//2]
                
                poses_control_knots_SE3 = poses_control_knots_SE3_mid.unsqueeze(0).repeat(self.num_views,1)
                poses_control_knots_se3 = poses_control_knots_SE3.Log()
                
                poses_noise_se3 = pp.randn_se3(self.num_views, sigma=initial_noise)
                poses_control_knots_se3 += poses_noise_se3
                
                self.pose_control_knots = pp.Parameter(poses_control_knots_se3)
            
        else:
            poses_control_knots_SE3 = pp.mat2SE3(camera_to_worlds, check=False) # NOTE: hack for now
            poses_control_knots_se3 = poses_control_knots_SE3.Log()
            poses_noise_se3 = pp.randn_se3(self.num_views, sigma=pose_noise)
            poses_control_knots_se3 += poses_noise_se3
            self.pose_control_knots = pp.Parameter(poses_control_knots_se3)

    def get_poses(self):
        if not self.pose_refine:
            if self.traj_mode == 'bezier':
                
                u = torch.linspace(start=0, end=1, steps=self.num_views, device=self.pose_control_knots.device)
                poses = bezier_interpolation(self.pose_control_knots.Exp(), u)
                poses = poses.matrix()[0] # (N, 4, 4)
                
            elif self.traj_mode == 'individual':
                
                poses = self.pose_control_knots.matrix()
                
        else:
            poses = self.pose_control_knots.matrix()

        return poses

class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    # test PoseOptModule
    camera_to_worlds = torch.eye(4).unsqueeze(0).repeat(10, 1, 1)

    pose_opt = PoseOptModule(camera_to_worlds, bezier_degree=5)

    poses = pose_opt.get_poses()

    print(poses)

    # Inspect all parameters
    print("\nParameters:")
    for name, param in pose_opt.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
