import bisect
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from fused_ssim import fused_ssim
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from nerfview import CameraState
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from dataloader import TimeLapseDataset, sun_angle
from gsplat_viewer import GsplatRenderTabState, GsplatViewer
from options import TimeSplattingConfig
from utils import ToneMapper, knn, set_random_seed


def create_splats_with_optimizers(
    dataset: TimeLapseDataset,
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    scene_scale: float = 1.0,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    use_shading: bool = False,
    use_weather: bool = False,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    points = torch.rand((init_num_pts, 3))
    points[..., 2] = 1
    points = init_extent * scene_scale * (points * 2 - 1)

    # Manually set 4:3 aspect ratio
    H = dataset[0]["image"].shape[0]
    W = dataset[0]["image"].shape[1]
    if H > W:  # height > width
        points[..., 0] *= W / H
    else:
        points[..., 1] *= H / W

    if use_shading:
        rgbs = torch.rand((init_num_pts, 1))
    else:
        rgbs = torch.rand((init_num_pts, 3))
    colors = torch.logit(rgbs)

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points[:, :2], 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)

    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    # Time parameters
    times = np.random.choice(dataset.dates, size=(N, 1))
    times = torch.tensor(times).float()

    if use_shading:
        sun_azimuth = np.random.choice(dataset.sun_angles[:, 0], size=(N, 1))
        sun_altitude = np.random.choice(dataset.sun_angles[:, 1], size=(N, 1))
        sun_angles = np.concatenate([sun_azimuth, sun_altitude], axis=-1)
        sun_angles = torch.tensor(sun_angles).float()
        if use_weather:
            weather = torch.tensor(np.random.choice(dataset.weather, size=(N, 1))).float()
            times = torch.cat([times, sun_angles, weather], dim=-1)  # [N, 4]
        else:
            times = torch.cat([times, sun_angles], dim=-1)  # [N, 4]

    time_scales = torch.log(
        torch.zeros_like(times) + torch.std(times, dim=0, keepdim=True) * 3
    )

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
        ("times", torch.nn.Parameter(times), means_lr * scene_scale),
        ("time_scales", torch.nn.Parameter(time_scales), scales_lr),
        ("colors", torch.nn.Parameter(colors), sh0_lr),
    ]

    if use_shading:
        d = times.shape[-1]  # number of time dimensions
        triu_len = d * (d - 1) // 2
        time_anisos = torch.zeros((N, triu_len), dtype=torch.float, device=device)

        params.append(("time_anisos", torch.nn.Parameter(time_anisos), quats_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: TimeSplattingConfig) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"
        cfg.result_dir = f"results/{os.path.basename(cfg.data_dir)}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        self.trainset = TimeLapseDataset(
            cfg.data_dir,
            split="train",
            data_factor=cfg.data_factor,
        )
        self.valset = TimeLapseDataset(cfg.data_dir, split="val")
        self.scene_scale = 1.1

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            dataset=self.trainset,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            scene_scale=self.scene_scale,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            use_shading=False,
            use_weather=False,
            device=self.device,
        )

        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if cfg.use_shading:
            self.shading_splats, self.shading_optimizers = (
                create_splats_with_optimizers(
                    dataset=self.trainset,
                    init_num_pts=cfg.init_num_pts,
                    init_extent=cfg.init_extent,
                    init_opacity=cfg.init_opa,
                    init_scale=cfg.init_scale,
                    means_lr=cfg.means_lr,
                    scales_lr=cfg.scales_lr,
                    opacities_lr=cfg.opacities_lr,
                    quats_lr=cfg.quats_lr,
                    sh0_lr=cfg.sh0_lr,
                    scene_scale=self.scene_scale,
                    sparse_grad=cfg.sparse_grad,
                    visible_adam=cfg.visible_adam,
                    use_shading=True,
                    use_weather=cfg.use_weather,
                    device=self.device,
                )
            )

            self.cfg.shading_strategy.check_sanity(
                self.shading_splats, self.shading_optimizers
            )

        print("Model initialized. Number of GS:", len(self.splats["means"]))

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        if cfg.use_shading:
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.shading_strategy_state = (
                    self.cfg.shading_strategy.initialize_state(
                        scene_scale=self.scene_scale
                    )
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.shading_strategy_state = (
                    self.cfg.shading_strategy.initialize_state()
                )
            else:
                assert_never(self.cfg.shading_strategy)

        self.app_optimizers = []
        if cfg.tone_mapper:
            self.tone_mapper = ToneMapper(3+1, len(self.trainset)).to(self.device)

            self.app_optimizers = [
                torch.optim.Adam(
                    self.tone_mapper.parameters(),
                    lr=cfg.tone_mapper_lr,
                ),
            ]

        self.bil_grid_optimizers = []
        if cfg.use_bilagrid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3,
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def splat_cholesky(self, splats):
        """
        We parameterize a N-dimensional covariance with the LDL decomposition.
        L is an unitriangular matrix with real values.
        """
        N = splats["times"].shape[0]
        d = splats["times"].shape[-1]
        tril = torch.tril_indices(d, d, offset=-1)
        L = torch.eye(d, device=self.device).reshape(1, d, d).repeat(N, 1, 1)
        if d > 1:
            L[:, tril[0], tril[1]] = splats["time_anisos"]

        return L

    def rasterize_splats(
        self,
        splats: Dict[str, Tensor],
        times: Tensor,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]

        # Time Splatting
        time_means = splats["times"]  # [N, d]
        time_scales = torch.exp(splats["time_scales"])  # [N, d]
        d = time_means.shape[-1]

        time_delta = times[None, ...] - time_means  # [N, d]

        # Upper triangular part of the time covariance matrix
        if d > 1:
            time_cholesky = self.splat_cholesky(splats).transpose(1, 2)  # [N, d, d]
            # Anisotropic component of the time covariance times the time deltas
            time_anisotropic = time_cholesky @ time_delta[..., None]  # [N, d, 1]
            time_anisotropic = time_anisotropic.squeeze(-1)  # [N, d]

        else:
            time_anisotropic = time_delta

        time_anisotropic = time_delta
        time_delta = time_anisotropic / time_scales  # [N, d]

        time_alpha = torch.exp(-0.5 * (time_delta * time_delta).sum(dim=-1))  # [N, 1]

        opacities = opacities * time_alpha

        colors = splats["colors"]  # [N, 3]
        colors = torch.sigmoid(colors)  # [N, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            camera_model="ortho",
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps))
        ]

        if cfg.use_bilagrid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]

            height, width = pixels.shape[1:3]

            times = (
                data["time"].float().to(device)
                + np.random.randn() * self.trainset.time_gap * cfg.time_noise_scale
            )

            # forward
            colors, _, info = self.rasterize_splats(
                splats=self.splats,
                times=times,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                near_plane=0.01,
                far_plane=100,
                render_mode="RGB",
                masks=masks,
                backgrounds=torch.tensor([(1.0, 1.0, 1.0)], device=self.device),
            )

            if cfg.use_shading:
                sun_angles = data["sun_angle"].float().to(device)
                sun_angles[:, 0] += (
                    np.random.randn()
                    * self.trainset.sun_angle_std[0].item()
                    * cfg.angle_noise_scale
                )
                sun_angles[:, 1] += (
                    np.random.randn()
                    * self.trainset.sun_angle_std[1].item()
                    * cfg.angle_noise_scale
                )

                if cfg.use_weather:
                    weather = data["weather"].float().to(device)
                    times = torch.cat([times, sun_angles[:, 0], sun_angles[:, 1], weather], dim=-1)  # [C, 4]
                else:
                    times = torch.cat([times, sun_angles[:, 0], sun_angles[:, 1]], dim=-1)  # [C, 3]

                shading_colors, _, shading_info = self.rasterize_splats(
                    splats=self.shading_splats,
                    times=times,
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    near_plane=0.01,
                    far_plane=100,
                    render_mode="RGB",
                    masks=masks,
                    backgrounds=torch.tensor([(1.0,)], device=self.device),
                )

                if cfg.tone_mapper:
                    I_wb = self.tone_mapper(times)
                    shading_colors = shading_colors * I_wb[None, ...]

                colors = colors * shading_colors

            alphas = data["alpha"].to(device) / 255
            colors = colors * alphas

            if cfg.use_bilagrid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            if cfg.use_shading:
                self.cfg.shading_strategy.step_pre_backward(
                    params=self.shading_splats,
                    optimizers=self.shading_optimizers,
                    state=self.shading_strategy_state,
                    step=step,
                    info=shading_info,
                )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.use_bilagrid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )

                if cfg.use_shading:
                    loss = (
                        loss
                        + cfg.opacity_reg
                        * torch.abs(
                            torch.sigmoid(self.shading_splats["opacities"])
                        ).mean()
                    )

            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

                if cfg.use_shading:
                    loss += (
                        +cfg.scale_reg
                        * torch.abs(torch.exp(self.shading_splats["scales"])).mean()
                    )

            loss.backward()

            desc = f"loss={loss.item():.3f}| "

            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                if cfg.use_bilagrid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}

                if cfg.use_shading:
                    data["shading_splats"] = self.shading_splats.state_dict()
                if cfg.tone_mapper:
                    data["tone_mapper"] = self.tone_mapper.state_dict()
                torch.save(data, f"{self.ckpt_dir}/ckpt_{step}.pt")

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

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if cfg.use_shading:
                for optimizer in self.shading_optimizers.values():
                    if cfg.visible_adam:
                        optimizer.step(visibility_mask)
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            if cfg.use_shading:
                # Run post-backward steps after backward and optimizer
                if isinstance(self.cfg.shading_strategy, DefaultStrategy):
                    self.cfg.shading_strategy.step_post_backward(
                        params=self.shading_splats,
                        optimizers=self.shading_optimizers,
                        state=self.shading_strategy_state,
                        step=step,
                        info=shading_info,
                        packed=cfg.packed,
                    )
                elif isinstance(self.cfg.shading_strategy, MCMCStrategy):
                    self.cfg.shading_strategy.step_post_backward(
                        params=self.shading_splats,
                        optimizers=self.shading_optimizers,
                        state=self.shading_strategy_state,
                        step=step,
                        info=shading_info,
                        lr=schedulers[0].get_last_lr()[0],
                    )
                else:
                    assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                # self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")

        # WARNING: Not tested with time splatting yet
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            times = data["time"].float().to(device)
            colors, _, _ = self.rasterize_splats(
                splats=self.splats,
                times=times,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                near_plane=0.01,
                far_plane=100,
                masks=masks,
                backgrounds=torch.tensor([(1.0, 1.0, 1.0)], device=self.device),
            )  # [1, H, W, 3]

            if cfg.use_shading:
                sun_angles = data["sun_angles"].float().to(device)

                if cfg.use_weather:
                    weather = data["weather"].float().to(device)
                    times = torch.cat([times, sun_angles[0], sun_angles[1], weather], dim=-1)
                else:
                    times = torch.cat([times, sun_angles[0], sun_angles[1]], dim=-1)

                shading_colors, shading_alphas, _ = self.rasterize_splats(
                    splats=self.shading_splats,
                    times=times,
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    near_plane=0.01,
                    far_plane=100,
                    masks=masks,
                    backgrounds=torch.tensor([(1.0,)], device=self.device),
                )
                colors = colors * shading_colors

            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                canvas,
            )

            pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors_p, pixels_p))
            metrics["ssim"].append(self.ssim(colors_p, pixels_p))
            metrics["lpips"].append(self.lpips(colors_p, pixels_p))
            if cfg.use_bilagrid:
                cc_colors = color_correct(colors, pixels)
                cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        ellipse_time /= len(valloader)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update(
            {
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
        )
        if cfg.use_bilagrid:
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
        else:
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
        # save stats as json
        with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"{stage}/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""

        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        times = np.linspace(0, 1, 600)
        # hours = np.linspace(0, 1, 600)

        period = 60  # seconds

        # hours = np.sin(hours * 2 * np.pi / period) * 4 + 12  # [8:00, 16:00] at 1Hz
        angles = []
        for i in range(len(times)):
            date, _, angle = self.abolute_to_relative_time(t=times[i], hour=10)
            # times[i] = t
            angles.append(angle)

        width, height = 1920, 1080

        fov = 120
        fx = fy = height / (2 * np.tan(np.deg2rad(fov) / 2))
        cy = height / 2
        cx = width / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        K = torch.from_numpy(K).float().to(self.device)
        c2w = torch.eye(4).float().to(device)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=60)
        for i in tqdm.trange(len(times), desc="Rendering trajectory"):
            albedos, _, _ = self.rasterize_splats(
                splats=self.splats,
                times=torch.tensor([times[i]]).float().cuda(),
                camtoworlds=c2w[None],
                Ks=K[None],
                width=width,
                height=height,
                near_plane=0.01,
                far_plane=100,
                render_mode="RGB",
                backgrounds=torch.tensor([(1.0, 1.0, 1.0)], device=self.device),
            )  # [1, H, W, 3]

            if cfg.use_shading:
                if cfg.use_weather:
                    weather = 1.
                    shading_time = torch.tensor([times[i], angles[i][0], angles[i][1], weather]).float().cuda()
                else:
                    shading_time = torch.tensor([times[i], angles[i][0], angles[i][1]]).float().cuda()

                shading, _, _ = self.rasterize_splats(
                    splats=self.shading_splats,
                    times=shading_time,
                    camtoworlds=c2w[None],
                    Ks=K[None],
                    width=width,
                    height=height,
                    near_plane=0.01,
                    far_plane=100,
                    backgrounds=torch.tensor([(1.0,)], device=self.device),
                    render_mode="RGB",
                )  # [1, H, W, 3]

                if self.cfg.tone_mapper:
                    I_wb = self.tone_mapper(shading_time)
                    shading = shading * I_wb[None, ...]
                else:
                    shading = shading.repeat(1, 1, 1, 3)

                renders = albedos * shading
                shadings = torch.clamp(shading, 0.0, 1.0)  # [1, H, W, 3]
            else:
                renders = albedos

            renders = torch.clamp(renders, 0.0, 1.0)
            albedos = torch.clamp(albedos, 0.0, 1.0)  # [1, H, W, 3]

            canvas_list = [renders]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def abolute_to_relative_time(self, t: float, hour: float):
        end_day: datetime = self.trainset.end_date
        start_day: datetime = self.trainset.start_date

        num_days = (end_day - start_day).days

        days_elapsed = int(num_days * t)
        date: datetime = start_day + timedelta(
            days=days_elapsed, seconds=hour * 60 * 60
        )
        left_idx = max(
            bisect.bisect_left(self.trainset.unique_days, days_elapsed) - 1, 0
        )
        right_idx = (
            left_idx + 1 if left_idx + 1 < len(self.trainset.unique_days) else left_idx
        )
        left = self.trainset.unique_days[left_idx]
        right = self.trainset.unique_days[right_idx]

        t = (days_elapsed - left) / (right - left + 1e-8)  # normalize to [0, 1]
        t = (
            self.trainset.days_linspace[left_idx] * (1 - t)
            + self.trainset.days_linspace[right_idx] * t
        )

        angle = sun_angle(date, self.trainset.gps['latitude'], self.trainset.gps['longitude'])
        angle = (angle[0] / 360, angle[1] / 90)  # normalize to [0, 1]

        return date, t, angle

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: GsplatRenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        # c2w = camera_state.c2w
        c2w = torch.eye(4).float().to(self.device)
        K = camera_state.get_K((width, height))
        K = torch.from_numpy(K).float().to(self.device)

        date, t, angle = self.abolute_to_relative_time(render_tab_state.time, render_tab_state.hour)

        times = torch.tensor([t]).float().cuda()

        render_colors, _, info = self.rasterize_splats(
            splats=self.splats,
            times=times,
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=100,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode="RGB",
            rasterize_mode=render_tab_state.rasterize_mode,
        )  # [1, H, W, 3]

        if self.cfg.use_shading and render_tab_state.render_mode != "albedo":
            if self.cfg.use_weather:
                times = (torch.tensor([render_tab_state.time, angle[0], angle[1], render_tab_state.weather]).float().cuda())
            else:
                times = (torch.tensor([render_tab_state.time, angle[0], angle[1]]).float().cuda())

            shading_colors, _, shading_info = self.rasterize_splats(
                splats=self.shading_splats,
                times=times,
                camtoworlds=c2w[None],
                Ks=K[None],
                width=width,
                height=height,
                near_plane=0.01,
                far_plane=100,
                backgrounds=torch.tensor([(1.0,)], device=self.device),
                render_mode="RGB",
                rasterize_mode=render_tab_state.rasterize_mode,
            )  # [1, H, W, 3]

            if self.cfg.tone_mapper:
                I_wb = self.tone_mapper(times)
                shading_colors = shading_colors * I_wb[None, ...]
            else:
                shading_colors = shading_colors.repeat(1, 1, 1, 3)

            if render_tab_state.render_mode == "full":
                render_colors = render_colors * shading_colors
            elif render_tab_state.render_mode == "shading":
                render_colors = shading_colors

        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode != "albedo":
            if render_tab_state.render_mode == "full" and self.cfg.use_shading:
                render_tab_state.total_gs_count += len(self.shading_splats["means"])
                render_tab_state.rendered_gs_count += (
                    (shading_info["radii"] > 0).all(-1).sum().item()
                )

            elif self.cfg.use_shading:
                render_tab_state.total_gs_count = len(self.shading_splats["means"])
                render_tab_state.rendered_gs_count = (
                    (shading_info["radii"] > 0).all(-1).sum().item()
                )

        render_tab_state.date = date.strftime("%Y-%m-%d %H:%M:%S")

        # colors represented with sh are not guranteed to be in [0, 1]
        render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
        renders = render_colors.cpu().numpy()

        return renders


def main(local_rank: int, world_rank: int, world_size: int, cfg: TimeSplattingConfig):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        for file in cfg.ckpt:
            ckpt = torch.load(file, map_location=runner.device, weights_only=True)

            for k in runner.splats.keys():
                runner.splats[k].data = ckpt["splats"][k]
    
            if cfg.use_shading:
                for k in runner.shading_splats.keys():
                    runner.shading_splats[k].data = ckpt["shading_splats"][k] 
                
                if cfg.tone_mapper:
                    runner.tone_mapper.load_state_dict(ckpt["tone_mapper"])

            step = ckpt["step"]

            # runner.eval(step=step)
            runner.render_traj(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            TimeSplattingConfig(
                strategy=DefaultStrategy(reset_every=100000, verbose=True),
                shading_strategy=DefaultStrategy(reset_every=100000, verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            TimeSplattingConfig(
                init_opa=0.5,
                init_scale=1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(cap_max=2_000_000, verbose=True),
                shading_strategy=MCMCStrategy(cap_max=2_000_000, verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilagrid:
        from fused_bilagrid import (
            BilateralGrid,
            color_correct,
            slice,
            total_variation_loss,
        )

    cli(main, cfg, verbose=True)
