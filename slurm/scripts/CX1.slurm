#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:ampere:1
#SBATCH --exclude=gpu[0,1,2,3,4,5,6,7,8]
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --job-name=x2ct-xray
#SBATCH --mem-per-cpu=28GB
#SBATCH -o slurm/logs/x2ct-xray.log
#SBATCH -t 03-00:00:00

python train_xray_vqgan.py --config configs/default_xray_vqgan_config.py
