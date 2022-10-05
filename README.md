## X2CT-VQVAE

### Setup
```
conda create --name x2ct-vqvae --file requirements.txt
conda activate x2ct-vqvae
```

Login to wandb with `wandb login`.

To train the diffusion model install flash attention from the instructions [here](https://github.com/HazyResearch/flash-attention). This requires access to nvcc, either installing cudnn through conda or activating the cuda module i.e. `module load cuda/11.6` should do this (using the version pytorch is compiled with).


### Training

Train X-Ray VQGAN with 
```
python train_xray_vqgan.py --config configs/default_xray_vqgan_config.py
```

Train CT VQGAN with
```
python train_ct_vqgan.py --config configs/default_ct_256_vqgan_config.py
```

Train Absorbing Diffusion sampler with
```
python train_sampler.py --config configs/default_absorbing_config.py
```
By default this sets up the X-Ray and CT VQGANs with the above configs, other configs can be passed in like
```
python train_sampler.py --config configs/default_absorbing_config.py --ct_config=<path to config> --xray_config=<path to config>
```
with values overwritten as described below e.g. `--ct_config.model.codebook_size=2048`

### Configs
Default config files are passed to each trainer as above. Values can be overwritten like this
```
python train_xray_vqgan.py --config configs/default_xray_vqgan_config.py --config.run.wandb_mode=disabled --config.data.num_xrays=4
```

In particular, each config has the variable `config.run.experiment` which sets the folder to store logs/checkpoints in.

All config values are 

