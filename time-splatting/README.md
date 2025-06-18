# Time Splatting

![teaser](../assets/time_splatting.jpg)


Time splatting is an approach to reconstruct a time-lapse, and to factorize it into intrinsic images. Time splatting was built with gsplat, and includes an interactive web viewer. 

This codebase was tested on PyTorch 2.7.1 and gsplat v1.5.2. 

## Dependencies
Please first install PyTorch, then install the rest of the dependencies.
```bash
pip install torch torchvision torchaudio
```
```bash
pip install gsplat pysolar pytz
```

## Training a Model
To train a model with time splatting, specify a densification strategy and a path to a time-lapse dataset. The default strategy is set to Markov Chain Monte Carlo (MCMC). 
```bash
python train.py mcmc --data_dir /path/to/data/dir
```

This implementation also includes many options to incorporate the latest splatting techniques for time splatting. By default, bilateral guided optimization is enabled, which helps sharpen effects such as shadows. 
