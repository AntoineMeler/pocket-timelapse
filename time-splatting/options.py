from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from gsplat.strategy import DefaultStrategy, MCMCStrategy
from typing_extensions import Literal


@dataclass
class DefaultConfig:
    """Default options for Gaussian Splatting."""

    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Path to the time-lapse dataset
    data_dir: str = "../../gsplat/examples/data/sunnyhoy_cropped_new"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Every N images there is a test image
    test_every: int = 8

    # Port for the viewer server
    port: int = 8080

    # Number of training steps
    max_steps: int = 50_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [500, 5000, 10000, 20000, 30000, 50000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [500, 5000, 10000, 20000, 30000, 50000])

    # Initial number of GSs.
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent.
    init_extent: float = 1.0
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(default_factory=MCMCStrategy)

    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = True
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"


@dataclass
class TimeSplattingConfig(DefaultConfig):
    """New Options for time splatting."""

    # Whether to use shading Gaussians for intrinsic image decomposition
    use_shading: bool = True
    use_weather: bool = True # antoine
    # Scaling factor for noise injected to the time label
    # Proportional to time difference between consecutive images
    time_noise_scale: float = 2.0
    # Scaling factor for noise injected to the sun angle label
    # Proportional to the sun angle std across images.
    angle_noise_scale: float = 0.1
    # Enable white balance optimization
    tone_mapper: bool = True
    tone_mapper_lr: float = 1e-3
    # Whether use fused-bilateral grid
    use_bilagrid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Strategy for shading Gaussians
    shading_strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=MCMCStrategy
    )
