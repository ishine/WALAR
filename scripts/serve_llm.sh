#!/bin/bash

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

base_model="Qwen2.5-3B-Instruct"
model_name="Qwen3-32B"
eval_type="da"

export CUDA_VISIBLE_DEVICES=0,1
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

python -m openrlhf.cli.serve_llm \
    --model_name  ${model_name}\
    --base_model $base_model \
    --port 6000 \
    --eval_type ${eval_type} \
    --max_len 2048 \
    --turns 1\
    --tensor_parallel_size ${num_gpus}

echo "${model_name} serves successfully!"