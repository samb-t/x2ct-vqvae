#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=gpu[0,1,2,3,4,5,6,7,8]
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --job-name=x2ct-sampler
#SBATCH --mem-per-cpu=28GB
#SBATCH -o slurm/logs/x2ct-sampler.log
#SBATCH -t 07-00:00:00

python train_sampler.py --config configs/default_sampler_chest_config.py --ct_config=configs/default_ct_vqgan_config.py --xray_config=configs/default_xray_vqgan_config.py
