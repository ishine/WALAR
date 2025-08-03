#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
year=24
model_name="Qwen3-32B-AWQ"
model_size="xxl"  ### model_size can be discarded if your model_name is not XComet or metricX
dtype="bf16"  ### dtype can be discarded if your model_name is metricX

# Only useful for Qwen
turns=1  
eval_type="da"

cd /mnt/gemini/data1/yifengliu/qe-lr/code

python evaluate_wmt.py \
  --wmt_year ${year}\
  --model_name ${model_name}\
  --model_size ${model_size}\
  --dtype ${dtype}\
  --eval_type ${eval_type} \
  --turns ${turns} \
  --output_dir /mnt/gemini/data1/yifengliu/qe-lr/output/wmt${year}

# CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 /mnt/data1/yifengliu/qe-lr/MetricX/evaluate_wmt.py \
#   --wmt_year 24 \
#   --model_name metricX \
#   --output_dir /mnt/data1/yifengliu/qe-lr/result/wmt24 \