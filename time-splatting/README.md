# Time Splatting

![teaser](../assets/time_splatting.jpg)


Time splatting is an approach to reconstruct a time-lapse, and to factorize it into intrinsic images. Time splatting was built with gsplat, and includes an interactive web viewer. 

This codebase was tested with Python 3.13, PyTorch 2.7.1 and gsplat v1.5.2. 

## Dependencies
Please first install PyTorch, then install the rest of the dependencies.
```bash
pip install torch torchvision
```
```bash
pip install -r requirements.txt --no-build-isolation
```

## Training a Model
To train a model with time splatting, specify a densification strategy (`default` or `mcmc`) and a path to a time-lapse dataset. 
```bash
python train.py mcmc --data_dir /path/to/data/dir
```

Since publication, we have integrated new techniques into the time splatting pipeline. The default strategy is now set to [Markov Chain Monte Carlo (MCMC)](https://doi.org/10.48550/arXiv.2404.09591). [Bilateral-guided optimization](https://doi.org/10.1145/3658148) is also supported, which helps sharpen shadows. However, it can oversmooth changes in geometry such as construction, so it is disabled by default. 

## Rendering a Model
To render a time-lapse from a trained checkpoint, run 
```bash
python train.py mcmc --data_dir /path/to/data/dir --ckpt /path/to/ckpt
```
This will also open the Viser viewer that you can access in your web browser.

## Choosing Hyperparameters
A full list of hyperparameters is in `options.py`. The hyperparameters for time splatting are
```python
class TimeSplattingConfig(DefaultConfig):
    # Whether to use shading Gaussians for intrinsic image decomposition
    use_shading: bool = True
    # Scaling factor for noise injected to the time label
    # Proportional to 1 / (days in time lapse)
    time_noise_scale: float = 2.0
    # Scaling factor for noise injected to the sun angle label
    # Proportional to the sun angle st. dev. across images.
    angle_noise_scale: float = 0.1
    # Enable white balance optimization
    tone_mapper: bool = False
    tone_mapper_lr: float = 1e-3
    # Whether use fused-bilateral grid
    use_bilagrid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
```

A good time-lapse should balance detail and smoothness. The most important hyperparameters controlling smoothness are `time_noise_scale` and `angle_noise_scale`, which are multipliers for the amount of noise injected into the time and sun angle labels. A higher value will create a smoother time-lapse, but will sacrifice detail. 

The rest of the hyperparameters in `DefaultConfig` are the same as in a typical 3D Gaussian splatting pipeline. These include options such as learning rate, initial number of Gaussians, etc. 

