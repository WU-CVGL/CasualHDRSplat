import json
import math
import os
import time
from typing import List, Literal, Optional

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from dataclasses import dataclass, field
from datasets.colmap import Dataset
from datasets.colmap_dataparser import ColmapParser
from datasets.deblur_nerf import DeblurNerfDataset
from datasets.hdr_deblur_nerf import HdrDeblurNerfDataset
from datasets.traj import generate_interpolated_path
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from nerfview.viewer import Viewer
from pose_viewer import PoseViewer
from simple_trainer import create_splats_with_optimizers, Runner
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModuleSE3,
    set_random_seed,
)
from badgs.camera_trajectory import CameraTrajectory, CameraTrajectoryConfig
from badgs.exposure_time_optimizer import ExposureTimeOptimizerConfig
from badgs.spline import SplineConfig
from badgs.spline_optimizer import SplineOptimizerConfig
from badgs.tonemapper import ToneMapper

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Visualize cameras in viewer
    visualize_cameras: bool = True

    # Path to the .pt file. If provide, it will skip training and render a video
    # ckpt: Optional[str] = "results/tanabata_mcmc_500k_grad25/ckpts/ckpt_29999.pt"
    ckpt: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    # data_dir: str = "data/360_v2/garden"
    # data_dir: str = "/datasets/bad-gaussian/data/bad-nerf-gtK-colmap-nvs/blurtanabata"
    # data_dir: str = "/datasets/HDR-Bad-Gaussian/images_ns_process"
    data_dir: str = "/datasets/HDR-Bad-Gaussian/HDRBlurcozyroom_colmap_true_no_blur"
    # Downsample factor for the dataset
    data_factor: int = 1
    # How much to scale the camera origins by. Default: 0.25 for LLFF scenes.
    scale_factor: float = 0.25
    # Directory to save results
    # result_dir: str = "results/garden_vanilla"
    # result_dir: str = "results/tanabata_vanilla"
    # result_dir: str = "results/tanabata_mcmc_500k_grad25"
    # result_dir: str = "results/tanabata_den4e-4_grad25_absgrad"
    # result_dir: str = "results/hdr_ikun_mcmc_500k_grad25_explr_1e-4"
    result_dir: str = "results/HDRBlurcozyroom_colmap_true_no_blur"
    # Every N images there is a test image
    test_every: int = 9999
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [500, 3_000, 7_000, 10_000, 15_000, 20_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [3_000, 7_000, 10_000, 15_000, 20_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    ###### MCMC parameters ######

    # whether to use MCMC
    enable_mcmc: bool = True
    # How much to scale the camera origins by
    scale_factor_mcmc: float = 1.0
    # MCMC Maximum number of GSs.
    cap_max: int = 100_000
    # MCMC samping noise learning rate
    noise_lr = 5e5
    # MCMC Opacity regularization
    opacity_reg = 0.01
    # MCMC Scale regularization
    scale_reg = 0.01
    # MCMC Initial opacity of GS
    init_opa_mcmc: float = 0.5
    # MCMC Initial scale of GS
    init_scale_mcmc: float = 0.1
    # Start refining GSs after this iteration
    refine_start_iter_mcmc: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter_mcmc: int = 25_000
    # Refine GSs every this steps
    refine_every_mcmc: int = 100

    ###### Non-MCMC parameters ######

    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 4e-3
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1
    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Refine GSs every this steps
    refine_every: int = 100
    # Reset opacities every this steps
    reset_every: int = 3000
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = True

    ########### Rasterization ###############

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    ########### Background ###############

    # Use random background for training to discourage transparency
    random_bkgd: bool = True

    ########### Appearance Opt ###############

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    ########### Depth Loss ###############

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    ########### Logging ###############

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    ########### Scale Reg ###############

    # If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians.
    use_scale_regularization: bool = True
    # threshold of ratio of gaussian max to min scale before applying regularization loss from the PhysGaussian paper
    max_gauss_ratio: float = 10.0

    ########### Motion Deblur ###############

    # BAD-Gaussians: Number of virtual cameras
    num_virtual_views: int = 10
    # BAD-Gaussians: Trajectory representation type
    traj_type: Literal["linear", "cubic"] = "cubic"
    # BAD-Gaussians: Trajectory interpolation ratio
    traj_interpolate_ratio: float = 2.0

    ########### Camera Opt ###############

    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-3
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Learning rate decay rate of camera optimization
    pose_opt_lr_decay: float = 1e-3
    # Initial noise for camera optimization
    pose_init_noise_se3: float = 1e-5
    # Pose gradient accumulation steps
    pose_gradient_accumulation_steps: int = 25

    ########### Novel View Eval Camera Opt ###############

    # Steps per image to evaluate the novel view synthesis
    nvs_steps: int = 500
    # Novel view synthesis evaluation pose learning rate
    nvs_pose_lr: float = 1e-3
    # Novel view synthesis evaluation pose regularization
    nvs_pose_reg: float = 0.0
    # Novel view synthesis evaluation pose learning rate decay
    nvs_pose_lr_decay: float = 1e-2

    ########### Exposure Time ###############

    # Whether to optimize exposure time as a parameter
    optimize_exposure_time: bool = True
    # Learning rate for exposure time optimization
    exposure_time_lr: float = 1e-4
    # Regularization for exposure time optimization as weight decay
    exposure_time_reg: float = 1e-6
    # Learning rate decay rate of exposure time optimization
    exposure_time_lr_decay: float = 1e-3
    # Initial noise for exposure time optimization
    exposure_time_init_noise: float = 1e-10
    # Exposure time gradient accumulation steps
    exposure_time_gradient_accumulation_steps: int = 25

    ########### Novel View Exposure Time ###############

    # Whether to optimize exposure time for novel view synthesis
    nvs_optimize_exposure_time: bool = False
    # Novel view synthesis evaluation exposure time learning rate
    nvs_exposure_time_lr: float = 1e-4
    # Novel view synthesis evaluation exposure time regularization
    nvs_exposure_time_reg: float = 0.0
    # Novel view synthesis evaluation exposure time learning rate decay
    nvs_exposure_time_lr_decay: float = 1e-3

    ########### HDR ###############

    # Read HDR Deblur-NeRF Dataset
    enable_hdr_deblur: bool = True

    ########### HDR Tone Mapping ###############

    use_HDR: bool = True
    k_times: float = 16.0
    tonemapper_lr: float = 0.005

    tonemapper_reg: float = 1e-6
    tonemapper_lr_decay: float = 0.01

    use_whitebalance: bool = False
    ######################################

    def __post_init__(self):
        # Change some correlated parameters after initialization of the config
        if self.enable_mcmc:
            print("MCMC enabled.")
            self.scale_factor = self.scale_factor_mcmc
            self.init_opa = self.init_opa_mcmc
            self.init_scale = self.init_scale_mcmc
            self.refine_start_iter = self.refine_start_iter_mcmc
            self.refine_stop_iter = self.refine_stop_iter_mcmc
            self.refine_every = self.refine_every_mcmc
        else:
            self.grow_grad2d = self.grow_grad2d / self.num_virtual_views

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)


class DeblurRunner(Runner):
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = ColmapParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            scale_factor=cfg.scale_factor,
        )
        if cfg.enable_hdr_deblur:
            self.parser.test_every = 0
            self.parser.valstart = 2
            self.parser.valend = 2
            self.trainset = HdrDeblurNerfDataset(self.parser, split="all")
            self.valset = HdrDeblurNerfDataset(self.parser, split="val")#novel view
            self.testset = HdrDeblurNerfDataset(self.parser, split="test")
        else:
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
            self.testset = DeblurNerfDataset(self.parser, split="test")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        if cfg.enable_mcmc:
            self.strategy = MCMCStrategy(
                verbose=True,
                cap_max=cfg.cap_max,
                noise_lr=cfg.noise_lr,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                refine_every=cfg.refine_every,
            )
            self.strategy.check_sanity(self.splats, self.optimizers)
            self.strategy_state = self.strategy.initialize_state()
        else:
            self.strategy = DefaultStrategy(
                verbose=True,
                scene_scale=self.scene_scale,
                prune_opa=cfg.prune_opa,
                grow_grad2d=cfg.grow_grad2d,
                grow_scale3d=cfg.grow_scale3d,
                prune_scale3d=cfg.prune_scale3d,
                # refine_scale2d_stop_iter=4000,  # splatfacto behavior
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                absgrad=cfg.absgrad,
                revised_opacity=cfg.revised_opacity,
            )
            self.strategy.check_sanity(self.splats, self.optimizers)
            self.strategy_state = self.strategy.initialize_state()

        # Camera Trajectory: Spline and Exposure Time Optimization
        if cfg.traj_type == "linear":
            spline_degree = 1
        elif cfg.traj_type == "cubic":
            spline_degree = 3
        else:
            raise NotImplementedError(f"Unknown trajectory type: {cfg.traj_type}")
        self.camera_trajectory_config: CameraTrajectoryConfig = CameraTrajectoryConfig(
            spline=SplineConfig(
                degree=spline_degree,
                spline_optimizer=SplineOptimizerConfig(
                    mode="SE3",
                    initial_noise_se3_std=cfg.pose_init_noise_se3,
                ),
            ),
            exposure_time_optimizer=ExposureTimeOptimizerConfig(
                mode="linear" if cfg.optimize_exposure_time else "off",
                initial_noise_std=cfg.exposure_time_init_noise,
            ),
            num_virtual_views=cfg.num_virtual_views,
            traj_interpolate_ratio=cfg.traj_interpolate_ratio,
        )
        self.camera_trajectory: CameraTrajectory = self.camera_trajectory_config.setup(
            timestamps=self.parser.timestamps,
            exposure_times=self.parser.exposure_times,
            c2ws=torch.tensor(self.parser.camtoworlds).float(),
            device="cuda",
        )
        camera_trajectory_param_groups = {}
        self.camera_trajectory.get_param_groups(camera_trajectory_param_groups)
        self.camera_trajectory_optimizer = torch.optim.Adam(
            camera_trajectory_param_groups["camera_opt"],
            lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
            weight_decay=cfg.pose_opt_reg,
        )
        self.exposure_time_optimizer = torch.optim.Adam(
            camera_trajectory_param_groups["exposure_time_opt"],
            lr=cfg.exposure_time_lr * math.sqrt(cfg.batch_size),
            weight_decay=cfg.exposure_time_reg,
        )

        # HDR Tone Mapper
        if cfg.use_HDR:
            self.tonemapper = ToneMapper(64).cuda()
            grad_vars = list(self.tonemapper.parameters())
            if cfg.use_whitebalance:
                self.tonemapper.setup_whitebalance(self.trainset.parser.image_paths,self.trainset.indices,self.device)
                grad_vars += list(self.tonemapper.wb.parameters())

            self.tonemapper_optimizer = torch.optim.Adam(
                grad_vars,
                lr=cfg.tonemapper_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.tonemapper_reg,
            )

        # Appearance Optimization
        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            if cfg.visualize_cameras:
                self.viewer = PoseViewer(
                    server=self.server,
                    render_fn=self._viewer_render_fn,
                    mode="training",
                )
            else:
                self.viewer = Viewer(
                    server=self.server,
                    render_fn=self._viewer_render_fn,
                    mode="training",
                )

        if not cfg.enable_mcmc:
            # Running stats for prunning & growing.
            n_gauss = len(self.splats["means"])
            self.running_stats = {
                "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
                "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
            }

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        # pose optimization has a learning rate schedule
        pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.camera_trajectory_optimizer,
            gamma=cfg.pose_opt_lr_decay ** (1.0 / max_steps)
        )
        schedulers.append(pose_scheduler)

        # exposure time optimization has a learning rate schedule
        if cfg.optimize_exposure_time:
            exposure_time_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.exposure_time_optimizer,
                gamma=cfg.exposure_time_lr_decay ** (1.0 / max_steps),
            )
            schedulers.append(exposure_time_scheduler)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        if cfg.visualize_cameras:
            self._init_viewer_state()

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            image_ids = data["image_id"]
            poses, exposure_times = self.camera_trajectory(image_ids, "uniform")
            camtoworlds = poses.matrix()
            num_cur_virt_views = camtoworlds.shape[0]
            Ks = Ks.tile((num_cur_virt_views, 1, 1))

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                exposure_time=exposure_times,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # BAD-Gaussians: average the virtual views
            colors = colors.mean(0)[None]

            if not cfg.enable_mcmc:
                self.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.enable_mcmc:
                loss = (
                        loss
                        + cfg.opacity_reg
                        * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
                loss = (
                        loss
                        + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            if cfg.use_scale_regularization and step % 10 == 0:
                scale_exp = torch.exp(self.splats["scales"])
                scale_reg = (
                        torch.maximum(
                            scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                            torch.tensor(cfg.max_gauss_ratio),
                        )
                        - cfg.max_gauss_ratio
                )
                scale_reg = 0.1 * scale_reg.mean()
                loss += scale_reg
            if cfg.use_HDR:
                loss += F.l1_loss(colors/2/colors.mean(), pixels/2/pixels.mean())
                if exposure_times < 0:
                    loss += -exposure_times
            else:
                scale_reg = torch.tensor(0.0).to(self.device)

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)

                # monitor pose optimization
                metrics_dict = {}
                self.camera_trajectory.get_metrics_dict(metrics_dict)
                self.writer.add_scalar(
                    "train/camera_opt_translation",
                    metrics_dict["camera_opt_translation"],
                    step
                )
                self.writer.add_scalar(
                    "train/camera_opt_rotation",
                    metrics_dict["camera_opt_rotation"],
                    step
                )
                # monitor exposure time
                if cfg.optimize_exposure_time:
                    self.writer.add_scalar(
                        "train/exposure_time_opt",
                        metrics_dict["exposure_time_opt"],
                        step
                    )
                    self.writer.add_scalar(
                        "train/exposureLR",
                        exposure_time_scheduler.get_last_lr()[0],
                        step
                    )

                # monitor pose learning rate
                self.writer.add_scalar("train/poseLR", pose_scheduler.get_last_lr()[0], step)
                # monitor ATE
                # self.visualize_traj(step)

                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)

                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            if cfg.enable_mcmc:
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize 3DGS
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # optimize appearance
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # optimize exposure time
            if cfg.optimize_exposure_time:
                if step % cfg.exposure_time_gradient_accumulation_steps == cfg.exposure_time_gradient_accumulation_steps - 1:
                    self.exposure_time_optimizer.step()
                if step % cfg.exposure_time_gradient_accumulation_steps == 0:
                    self.exposure_time_optimizer.zero_grad(set_to_none=True)

            # optimize tone mapper
            if cfg.use_HDR:
                self.tonemapper_optimizer.step()
                self.tonemapper_optimizer.zero_grad(set_to_none=True)


            # optimize camera trajectory
            if step % cfg.pose_gradient_accumulation_steps == cfg.pose_gradient_accumulation_steps - 1:
                self.camera_trajectory_optimizer.step()
            if step % cfg.pose_gradient_accumulation_steps == 0:
                self.camera_trajectory_optimizer.zero_grad(set_to_none=True)

            # update learning rate
            for scheduler in schedulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                        "camera_trajectory": self.camera_trajectory.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval_deblur(step)
                if self.valset is not None:
                    self.eval_novel_view(step)
                # self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    # def order_loss(self):
    #     self.tonemapper

    @torch.no_grad()
    def eval_deblur(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(testloader):
            camtoworlds_gt = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            image_ids = data["image_id"]
            pose, exposure_time = self.camera_trajectory(image_ids, "start")
            camtoworlds = pose.matrix()[None, ...]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                exposure_time=exposure_time,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/{step:04d}_deblur_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(testloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"Deblurring at Step_{step:04d}:"
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means"]),
        }
        with open(f"{self.stats_dir}/deblur_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"deblur/{k}", v, step)
        self.writer.flush()

    def eval_novel_view(self, step: int):
        """Entry for evaluation."""
        NVS_DEBUG_STEP = 50
        DEBUG_PRINT_PRECISION = 10
        torch.set_printoptions(precision=DEBUG_PRINT_PRECISION)
        ZERO_INT = torch.tensor([0])

        cfg = self.cfg
        device = self.device
        print("Running evaluation...")

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )

        # Freeze the scene
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = False

        if cfg.use_HDR:
            for param_group in self.tonemapper_optimizer.param_groups:
                param_group["params"][0].requires_grad = False
            for param_group in self.exposure_time_optimizer.param_groups:
                param_group["params"][0].requires_grad = False

        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = pixels.shape[1:3]
            image_ids = data["image_id"].to(device)
            exposure_time = data['exposure_time'].to(device)
            print('NVS eval - Initial exposure time:', exposure_time.item())
            pixels_ = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]

            novel_view_schedulers = []

            # Exposure Time Optimization
            novel_view_exposure_time_adjust = ExposureTimeOptimizerConfig(
                mode="linear" if cfg.nvs_optimize_exposure_time else "off",
                initial_noise_std=cfg.exposure_time_init_noise,
            ).setup(
                num_cameras=1,
                device=device,
            )
            if cfg.nvs_optimize_exposure_time:
                novel_view_exposure_time_optimizer = torch.optim.Adam(
                    novel_view_exposure_time_adjust.parameters(),
                    lr=cfg.nvs_exposure_time_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.nvs_exposure_time_reg,
                    eps=1e-15,
                )
                novel_view_exposure_time_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    novel_view_exposure_time_optimizer,
                    gamma=cfg.nvs_exposure_time_lr_decay ** (1.0 / cfg.max_steps)
                )
                novel_view_schedulers.append(novel_view_exposure_time_scheduler)

            # Pose Optimization
            novel_view_pose_adjust = CameraOptModuleSE3(1).to(self.device)
            novel_view_pose_adjust.random_init(cfg.pose_init_noise_se3)
            novel_view_pose_optimizer = torch.optim.Adam(
                novel_view_pose_adjust.parameters(),
                lr=cfg.nvs_pose_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.nvs_pose_reg,
                eps=1e-15,
            )
            novel_view_pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                novel_view_pose_optimizer,
                gamma=cfg.pose_opt_lr_decay ** (1.0 / cfg.max_steps)
            )
            novel_view_schedulers.append(novel_view_pose_scheduler)

            for j in range(cfg.nvs_steps):
                camtoworlds_new = novel_view_pose_adjust(camtoworlds, torch.tensor(ZERO_INT).to(self.device))
                exposure_time_new = exposure_time + novel_view_exposure_time_adjust(ZERO_INT)
                colors, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds_new,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB",
                    exposure_time=exposure_time_new,
                )
                colors_ = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]

                # loss
                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - self.ssim(colors_, pixels_)
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

                loss.backward()

                if cfg.nvs_optimize_exposure_time:
                    novel_view_exposure_time_optimizer.step()
                    novel_view_exposure_time_optimizer.zero_grad(set_to_none=True)
                novel_view_pose_optimizer.step()
                novel_view_pose_optimizer.zero_grad(set_to_none=True)

                for scheduler in novel_view_schedulers:
                    scheduler.step()

                with torch.no_grad():
                    if j % NVS_DEBUG_STEP == 0:
                        colors_ = torch.clamp(colors_, 0.0, 1.0)
                        psnr = self.psnr(colors_.detach(), pixels_)
                        ssim = self.ssim(colors_, pixels_)
                        lpips = self.lpips(colors_.detach(), pixels_)
                        print(
                            f"NVS eval at Step_{step:04d}:"
                            f"NVS_IMG_#{i:04d}_step_{j:04d}:"
                            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                            f"exposure_time:{exposure_time_new.item():.5f}"
                        )
                        metrics["psnr"].append(psnr)
                        metrics["ssim"].append(ssim)
                        metrics["lpips"].append(lpips)

                        # # NVS Debugging
                        # stats = {
                        #     "psnr": psnr.item(),
                        #     "ssim": ssim.item(),
                        #     "lpips": lpips.item(),
                        # }
                        # for k, v in stats.items():
                        #     self.writer.add_scalar(f"nvs/{step}/{i}/{k}", v, j)
                        # self.writer.add_scalar(f"nvs/{step}/{i}/pose_lr", scheduler.get_last_lr()[0], j)
                        # self.writer.add_scalar(f"nvs/{step}/{i}/camera_opt_translation", novel_view_pose_adjust.poses_opt[:, :3].mean(), j)
                        # self.writer.add_scalar(f"nvs/{step}/{i}/camera_opt_rotation", novel_view_pose_adjust.poses_opt[:, 3:].mean(), j)
                        # self.writer.flush()

                        # write images
                        canvas = torch.cat([pixels, colors], dim=2).squeeze(0).detach().cpu().numpy()
                        imageio.imwrite(
                            f"{self.render_dir}/{step:04d}_nvs_{i:04d}_{j:04d}.png", (canvas * 255).astype(np.uint8)
                        )
        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
        }
        with open(f"{self.stats_dir}/nvs_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"nvs/{k}", v, step)
        self.writer.flush()

        # Unfreeze the scene
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = True
        if cfg.use_HDR:
            for param_group in self.tonemapper_optimizer.param_groups:
                param_group["params"][0].requires_grad = True
            for param_group in self.exposure_time_optimizer.param_groups:
                param_group["params"][0].requires_grad = True

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i: i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                exposure_time=torch.Tensor(0)
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30, codec='libx264')
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        if not cfg.disable_viewer:
            assert self.viewer and self.trainset
            self.viewer.init_scene(train_dataset=self.trainset, train_state="training")


def main(cfg: Config):
    runner = DeblurRunner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k].detach().to(runner.device)
            runner.camera_trajectory.exposure_time_optimizer.adjustment = ckpt[
                "camera_trajectory"]["exposure_time_optimizer.adjustment"].detach().to(runner.device)
            runner.camera_trajectory.spline.spline_optimizer.pose_adjustment = ckpt[
                "camera_trajectory"]["'spline.spline_optimizer.pose_adjustment'"].detach().to(runner.device)
        runner.eval_deblur(step=ckpt["step"])
        if runner.valset is not None:
            runner.eval_novel_view(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
