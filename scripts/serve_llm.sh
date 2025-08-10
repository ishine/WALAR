#!/bin/bash

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

base_model="Qwen3-4B"
model_name="Qwen3-30B-A3B"
eval_type="da"

export CUDA_VISIBLE_DEVICES=5,6
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

python -m openrlhf.cli.serve_llm \
    --model_name  ${model_name}\
    --base_model $base_model \
    --port 5555 \
    --eval_type ${eval_type} \
    --max_len 2048 \
    --turns 1\
    --lang_detect \
    --tensor_parallel_size ${num_gpus}

echo "${model_name} serves successfully!"