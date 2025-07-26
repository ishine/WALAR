#!/bin/bash
#SBATCH --job-name=serve_models
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2            # You might adjust this depending on your cluster
#SBATCH --mem=32G
#SBATCH --output=/mnt/gemini/data1/yifengliu/qe-lr/logs/serve_%j_%t.out
#SBATCH --error=/mnt/gemini/data1/yifengliu/qe-lr/logs/serve_%j_%t.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=yifengl@andrew.cmu.edu
#SBATCH --partition=aries

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl


# Assign model_name, port, batch_size, and GPU based on task id
case "$SLURM_PROCID" in
  0)
    export CUDA_VISIBLE_DEVICES=0
    MODEL_NAME="metricX"
    PORT=5000
    BATCH_SIZE=8
    ;;
  1)
    export CUDA_VISIBLE_DEVICES=1
    MODEL_NAME="XComet"
    PORT=4000
    BATCH_SIZE=8
    ;;
  2)
    export CUDA_VISIBLE_DEVICES=2
    MODEL_NAME="Comet22"
    PORT=7000
    BATCH_SIZE=16
    ;;
esac

echo $SLURM_PROCID

echo "Starting $MODEL_NAME on port $PORT with batch size $BATCH_SIZE on GPU $CUDA_VISIBLE_DEVICES"

cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

python -m openrlhf.cli.serve_rm \
    --model_name $MODEL_NAME \
    --port $PORT \
    --max_len 1536 \
    --batch_size $BATCH_SIZE
