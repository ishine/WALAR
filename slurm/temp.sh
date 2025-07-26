#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2            # You might adjust this depending on your cluster
#SBATCH --mem=32G
#SBATCH --output=/mnt/gemini/data1/yifengliu/qe-lr/logs/temp_%j.out
#SBATCH --error=/mnt/gemini/data1/yifengliu/qe-lr/logs/temp_%j.err
#SBATCH --partition=taurus

# project settings
echo "CUDA gpus: $CUDA_VISIBLE_DEVICES"

sleep infinity

# srun --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus=1 --partition=taurus --account=yifengliu --pty bash -i