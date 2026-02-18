export CUDA_VISIBLE_DEVICES=4,5,6,7
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${num_gpus}


CONDA_PATH=/mnt/gemini/data1/yifengliu/miniconda3
OPENRLHF_PATH=/mnt/gemini/data1/yifengliu/qe-lr/openrlhf
eval "$(${CONDA_PATH}/bin/conda shell.bash hook)"
which python
source ${CONDA_PATH}/bin/activate qe-rl
# export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=1200

cd $OPENRLHF_PATH

export HF_HOME=/mnt/gemini/data2/yifengliu/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/datasets
export HF_HUB_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/hub
export DS_SKIP_CUDA_CHECK=1
export RAY_DEBUG_POST_MORTEM=1

wandb_token=5bebcc325992863eb55622d9ad2e7c85c95a1f15
# cmu key
# wandb_token=e00b93c51b52fed0712d2130a4df508e9a41e95c

declare -A path_dict
path_dict["Llama"]="/mnt/gemini/data1/yifengliu/model/Llama-3.2-3B-Instruct"
path_dict["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-4B-Instruct-2507"
path_dict["LlamaX"]="/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca"

model="Qwen"
dataname="final_qwen_mix250"
size="4B"
reward_name="alpha20"

# remote_rm_url
# remote_rm_url2
# remote_comet_url
# remote_metric_reference_url

# ${path_dict[$model]}
# num_gpus=4
#--remote_comet_url http://localhost:4000/get_reward \
# --pretrain /mnt/gemini/data1/yifengliu/model/Qwen${version}-${size} \
# --ckpt_path /mnt/gemini/data1/yifengliu/checkpoints/${reward_name}-${model}${version}-${size}-${dataname}-1M-bsz128 \
# --colocate_actor_ref
# colocate_all_models
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/mnt/gemini/data1/yifengliu/qe-lr/openrlhf"}' \
    -- python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node ${num_gpus} \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node ${num_gpus} \
    --vllm_num_engines ${num_gpus} \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_sync_backend nccl \
    --ref_reward_offload \
    --pretrain ${path_dict[$model]} \
    --remote_rm_url http://localhost:2000/get_reward \
    --remote_comet_url http://localhost:5555/get_reward \
    --micro_train_batch_size 32 \
    --train_batch_size 1024 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 128 \
    --n_samples_per_prompt 8 \
    --enable_prefix_caching \
    --max_samples 10000000 \
    --max_epochs 1 \
    --prompt_max_len 512 \
    --generate_max_len 512 \
    --packing_samples \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --use_kl_loss \
    --init_kl_coef 0.01 \
    --kl_estimator k3 \
    --advantage_estimator group_norm \
    --prompt_data /mnt/gemini/data1/yifengliu/qe-lr/data/train/${dataname}.jsonl \
    --eval_dir "/mnt/gemini/data1/yifengliu/data/flores101_dataset/dev" \
    --eval_temperature 0.0 \
    --eval_steps 100000 \
    --eval_n_samples_per_prompt 1\
    --input_key input_key \
    --apply_chat_template \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --overlap_comm \
    --gradient_checkpointing \
    --temperature 1 \
    --save_steps 100 \
    --save_path /mnt/gemini/data1/yifengliu/checkpoints/final/${reward_name}-${model}-${size}-${dataname}-bsz1024 \
    --ckpt_path /mnt/gemini/data1/yifengliu/checkpoints/${reward_name}-${model}-${size}-${dataname}-bsz1024 \
    --load_checkpoint \
    --save_hf_ckpt \
    --use_wandb ${wandb_token}\
    --wandb_run_name "${reward_name}-${model}-${size}-${dataname}" \
    --enforce_eager \
    --deepspeed_enable_sleep \
    --vllm_enable_sleep


# export CUDA_VISIBLE_DEVICES=4,5,6,7
# num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
# ray start --head --node-ip-address 0.0.0.0 --num-gpus ${num_gpus}

# eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
# which python
# source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl
# # export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=1200

# cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

# export HF_HOME=/mnt/gemini/data2/yifengliu/.cache/huggingface
# export TRANSFORMERS_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/transformers
# export HF_DATASETS_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/datasets
# export HF_HUB_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/hub
# export DS_SKIP_CUDA_CHECK=1
# export RAY_DEBUG_POST_MORTEM=1

# # wandb_token=5bebcc325992863eb55622d9ad2e7c85c95a1f15
# # cmu key
# wandb_token=e00b93c51b52fed0712d2130a4df508e9a41e95c

# declare -A path_dict
# path_dict["Llama"]="/mnt/gemini/data1/yifengliu/model/Llama-3.2-3B-Instruct"
# path_dict["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
# path_dict["LlamaX"]="/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca"

# model="Qwen"
# src="en"
# tgt="mix-mid2"
# dataname="eight_direction41_mix-1m"
# version="3"
# size="4B"
# reward_name="eight_directions"
# if [ "${#tgt}" -le 3 ]; then
#     evaluation_step=10
# else
#     evaluation_step=100000
# fi
# # remote_rm_url
# # remote_rm_url2
# # remote_comet_url
# # remote_metric_reference_url

# #--remote_comet_url http://localhost:4000/get_reward \
# # --pretrain /mnt/gemini/data1/yifengliu/model/Qwen${version}-${size} \
# # --ckpt_path /mnt/gemini/data1/yifengliu/checkpoints/${reward_name}-${model}${version}-${size}-${dataname}-1M-bsz128 \
# ray job submit --address="http://127.0.0.1:8265" \
#     --runtime-env-json='{"working_dir": "/mnt/gemini/data1/yifengliu/qe-lr/openrlhf", "excludes": ["/mnt/gemini/data1/yifengliu/qe-lr/openrlhf/wandb/run-20250726_165454-yl7o7sbx/run-yl7o7sbx.wandb"]}' \
#     -- python -m openrlhf.cli.train_ppo_ray \
#     --ref_num_nodes 1 \
#     --ref_num_gpus_per_node ${num_gpus} \
#     --actor_num_nodes 1 \
#     --actor_num_gpus_per_node ${num_gpus} \
#     --vllm_num_engines ${num_gpus} \
#     --vllm_tensor_parallel_size 1 \
#     --colocate_all_models \
#     --vllm_gpu_memory_utilization 0.7 \
#     --ref_reward_offload \
#     --pretrain /mnt/gemini/data1/yifengliu/checkpoints/Final-Qwen3-4B-final_mix-160k-1M-bsz128/global_step1000_hf \
#     --remote_rm_url http://localhost:2000/get_reward \
#     --remote_comet_url http://localhost:5555/get_reward \
#     --micro_train_batch_size 16 \
#     --train_batch_size 64 \
#     --micro_rollout_batch_size 16 \
#     --rollout_batch_size 128 \
#     --n_samples_per_prompt 8 \
#     --max_samples 250000 \
#     --max_epochs 1 \
#     --prompt_max_len 2048 \
#     --generate_max_len 2048 \
#     --packing_samples \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 5e-7 \
#     --use_kl_loss \
#     --init_kl_coef 0.01 \
#     --kl_estimator k3 \
#     --advantage_estimator group_norm \
#     --prompt_data /mnt/gemini/data1/yifengliu/qe-lr/data/train/${dataname}.jsonl \
#     --src ${src} \
#     --tgt ${tgt} \
#     --eval_dir "/mnt/gemini/data1/yifengliu/data/flores101_dataset/dev" \
#     --eval_temperature 0.0 \
#     --eval_steps 100000 \
#     --eval_n_samples_per_prompt 1\
#     --input_key input_key \
#     --apply_chat_template \
#     --normalize_reward \
#     --ring_attn_size 4 \
#     --ring_head_stride 4 \
#     --adam_offload \
#     --overlap_comm \
#     --gradient_checkpointing \
#     --temperature 1 \
#     --save_steps 50 \
#     --save_path /mnt/gemini/data1/yifengliu/checkpoints/final/${reward_name}-${model}${version}-${size}-${dataname}-1M-bsz128 \
#     --ckpt_path /mnt/gemini/data1/yifengliu/checkpoints/${reward_name}-${model}${version}-${size}-${dataname}-1M-bsz128 \
#     --load_checkpoint \
#     --save_hf_ckpt \
#     --use_wandb ${wandb_token}\
#     --wandb_run_name "${reward_name}-${model}${version}-${size}-${dataname}-bsz128" \
#     --enforce_eager \
#     --vllm_enable_sleep \
#     --deepspeed_enable_sleep

