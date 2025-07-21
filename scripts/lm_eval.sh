export CUDA_VISIBLE_DEVICES=7

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

model_path="/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-zh-1M-bsz128/global_step120_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step210_hf
# /mnt/gemini/data1/yifengliu/model/Qwen2.5-3B-Instruct

lm_eval --model vllm \
    --model_args pretrained=${model_path},tensor_parallel_size=${num_gpus},dtype="auto",gpu_memory_utilization=0.8,enable_thinking=False \
    --tasks mgsm_direct \
    --output_path /mnt/gemini/data1/yifengliu/qe-lr/output/mgsm \
    --log_samples \
    --batch_size auto
