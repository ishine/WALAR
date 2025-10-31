export CUDA_VISIBLE_DEVICES=4
declare -A model_path
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-8B"
model_path["Qwen3X"]="/mnt/gemini/data1/yifengliu/model/Qwen3-XPlus-8B"
model_path["llama"]="/mnt/gemini/data1/yifengliu/model/Llama-3.2-3B-Instruct"
model_path["llamax"]="/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule_reward-LlamaX3-8B-schedule_no_pl-1M-bsz128/global_step550_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule_mix-LlamaX3-8B-schedule_mix-1M-bsz128/global_step1400_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule_mix-LlamaX3-8B-schedule_mix-1M-bsz128/global_step250_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule_no_pl-LlamaX3-8B-schedule_no_pl-1M-bsz128/global_step50_hf"
model_path["Qwen-base"]="/mnt/gemini/data1/yifengliu/model/Qwen3-4B-Base"


model_name="checkpoint"
final_path=${model_path[$model_name]}

cd /mnt/gemini/data1/yifengliu/BenchMAX

lm-eval -m vllm --model_args pretrained=${final_path} --tasks xifeval_multi --batch_size auto --apply_chat_template --include_path tasks/ifeval --log_samples -o results/ifeval

# lm-eval -m vllm --model_args pretrained=${final_path} --tasks xmgsm_native_cot_multi --batch_size auto --apply_chat_template --include_path tasks/mgsm --log_samples -o results/mgsm

# lm-eval -m vllm --model_args pretrained=${final_path} --tasks xgpqa_main_native_cot_zeroshot_multi --batch_size auto --apply_chat_template --include_path tasks/gpqa --log_samples -o results/gpqa