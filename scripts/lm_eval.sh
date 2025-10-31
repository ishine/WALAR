export CUDA_VISIBLE_DEVICES=6

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl


model_path="/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
# model_path="/mnt/gemini/data1/yifengliu/checkpoints/final/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128"
# model_path="/mnt/gemini/data1/yifengliu/model/Llama-3.2-3B-Instruct"
# model_path="/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca"
# model_path="/mnt/gemini/data1/yifengliu/checkpoints/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"
# model_path="/mnt/gemini/data1/yifengliu/checkpoints/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128"
# model_path="/mnt/gemini/data1/yifengliu/checkpoints/schedule_reward-LlamaX3-8B-schedule_no_pl-1M-bsz128/global_step550_hf"
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

# Rule-Detect-MetricX-Qwen3-4B-en-zh-1M-bsz128
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-zh-1M-bsz128/global_step120_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step210_hf
#"/mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-zh-1M-bsz128/global_step460_hf"
# /mnt/gemini/data1/yifengliu/model/Qwen2.5-3B-Instruct

# mgsm_direct, mmlu_prox_zh, mmlu_pro

# xcopa, xnli, xstorycloze, xwinograd
# task_name="xstorycloze"
# lm_eval --model vllm \
#     --model_args pretrained=${model_path} \
#     --tasks ${task_name} \
#     --apply_chat_template \
#     --log_samples -o /mnt/gemini/data1/yifengliu/BenchMAX/results/${task_name} \
#     --batch_size auto

task_name=mmlu_prox_en
lm_eval --model vllm \
    --model_args pretrained=${model_path} \
    --tasks ${task_name} \
    --apply_chat_template \
    --log_samples -o /mnt/gemini/data1/yifengliu/output/mmlu_prox/zh \
    --batch_size auto