# Time Splatting

![teaser](../assets/time_splatting.jpg)


Time splatting is an approach to reconstruct a time-lapse, and to factorize it into intrinsic images. Time splatting was built with gsplat, and includes an interactive web viewer. 

This codebase was tested with PyTorch 2.7.1 and gsplat v1.5.2. 

## Dependencies
Please first install PyTorch, then install the rest of the dependencies.
```bash
pip install torch torchvision torchaudio
```
```bash
pip install gsplat pysolar pytz
```

## Training a Model
To train a model with time splatting, specify a densification strategy and a path to a time-lapse dataset. The default strategy is set to [Markov Chain Monte Carlo (MCMC)](https://doi.org/10.48550/arXiv.2404.09591). 
```bash
python train.py mcmc --data_dir /path/to/data/dir
```

By default, [bilateral-guided optimization](https://doi.org/10.1145/3658148) is enabled, which helps sharpen effects such as shadows. 
