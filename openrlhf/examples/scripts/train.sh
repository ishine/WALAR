export CUDA_VISIBLE_DEVICES=4,5,6,7
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl
# export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=1200

cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

export HF_HOME=/mnt/gemini/data2/yifengliu/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/datasets
export HF_HUB_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/hub
export DS_SKIP_CUDA_CHECK=1
export RAY_DEBUG_POST_MORTEM=1

# wandb_token=5bebcc325992863eb55622d9ad2e7c85c95a1f15
# cmu key
wandb_token=e00b93c51b52fed0712d2130a4df508e9a41e95c

src="en"
tgt="mix-mid2"
version="3"
size="4B"
reward_name="Test"
if [ "${#tgt}" -le 3]; then
    evaluation_step=10
else
    evaluation_step=100000
fi
# remote_rm_url
# remote_rm_url2
# remote_comet_url
# remote_metric_reference_url

#--remote_comet_url http://localhost:4000/get_reward \
# --pretrain /mnt/gemini/data1/yifengliu/model/Qwen${version}-${size} \
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/mnt/gemini/data1/yifengliu/qe-lr/openrlhf", "excludes": ["/mnt/gemini/data1/yifengliu/qe-lr/openrlhf/wandb/run-20250726_165454-yl7o7sbx/run-yl7o7sbx.wandb"]}' \
    -- python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.7 \
    --ref_reward_offload \
    --pretrain /mnt/gemini/data1/yifengliu/model/Qwen3-4B \
    --remote_rm_url http://localhost:2000/get_reward \
    --remote_comet_url http://localhost:5555/get_reward \
    --micro_train_batch_size 16 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 128 \
    --n_samples_per_prompt 8 \
    --max_samples 100000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --packing_samples \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --use_kl_loss \
    --init_kl_coef 0.01 \
    --kl_estimator k3 \
    --advantage_estimator group_norm \
    --prompt_data /mnt/gemini/data1/yifengliu/qe-lr/data/train/3base_${src}-${tgt}-1m.jsonl \
    --src ${src} \
    --tgt ${tgt} \
    --eval_dir "/mnt/gemini/data1/yifengliu/data/flores101_dataset/dev" \
    --eval_temperature 0.0 \
    --eval_steps 10000 \
    --eval_n_samples_per_prompt 1\
    --input_key input_key \
    --apply_chat_template \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --overlap_comm \
    --gradient_checkpointing \
    --temperature 1 \
    --save_steps 20 \
    --save_path /mnt/gemini/data1/yifengliu/checkpoints/final/${reward_name}-Qwen${version}-${size}-${src}-${tgt}-1M-bsz128 \
    --ckpt_path /mnt/gemini/data1/yifengliu/checkpoints/${reward_name}-Qwen${version}-${size}-${src}-${tgt}-1M-bsz128 \
    --load_checkpoint \
    --save_hf_ckpt \
    --use_wandb ${wandb_token}\
    --wandb_run_name "${reward_name}-Qwen${version}-${size}-${src}-${tgt}-1M-bsz128" \
    --enforce_eager \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# # ray start --head --node-ip-address 0.0.0.0 --num-gpus 4

# eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
# which python
# source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl
# # export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=1200

# cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

# export DS_SKIP_CUDA_CHECK=1
# wandb_token=5bebcc325992863eb55622d9ad2e7c85c95a1f15

# ray job submit --address="http://127.0.0.1:8265" \
#     --runtime-env-json='{"working_dir": "/mnt/gemini/data1/yifengliu/qe-lr/openrlhf"}' \
#     -- python -m openrlhf.cli.train_ppo_ray \
#     --ref_num_nodes 1 \
#     --ref_num_gpus_per_node 2 \
#     --actor_num_nodes 1 \
#     --actor_num_gpus_per_node 2 \
#     --vllm_num_engines 1 \
#     --vllm_tensor_parallel_size 1 \
#     --colocate_actor_ref \
#     --ref_reward_offload \
#     --pretrain /mnt/gemini/data1/yifengliu/model/Qwen2.5-3B-Instruct \
#     --remote_rm_url http://localhost:5000/get_reward \
#     --remote_comet_url http://localhost:3000/get_reward \
#     --remote_metric_reference_url http://localhost:4000/get_reward \
#     --micro_train_batch_size 16 \
#     --train_batch_size 128 \
#     --micro_rollout_batch_size 16 \
#     --rollout_batch_size 128 \
#     --n_samples_per_prompt 8 \
#     --max_samples 100000 \
#     --max_epochs 1 \
#     --prompt_max_len 1024 \
#     --generate_max_len 2048 \
#     --packing_samples \
#     --zero_stage 2 \
#     --bf16 \
#     --actor_learning_rate 5e-7 \
#     --use_kl_loss \
#     --init_kl_coef 0.01 \
#     --kl_estimator k3 \
#     --advantage_estimator group_norm \
#     --prompt_data /mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-zh-1m.jsonl \
#     --src "en" \
#     --tgt "zh" \
#     --eval_dir "/mnt/gemini/data1/yifengliu/data/flores101_dataset/dev" \
#     --eval_temperature 0.0 \
#     --eval_steps 10 \
#     --eval_n_samples_per_prompt 1\
#     --input_key input_key \
#     --apply_chat_template \
#     --normalize_reward \
#     --adam_offload \
#     --flash_attn \
#     --gradient_checkpointing \
#     --temperature 1 \
#     --save_steps 10 \
#     --save_path /mnt/gemini/data1/yifengliu/checkpoints/final/Qwen2.5-3B-Instruct-En-Zh-1M-2 \
#     --ckpt_path /mnt/gemini/data1/yifengliu/checkpoints/Qwen2.5-3B-Instruct-En-Zh-1M-2 \
#     --load_checkpoint \
#     --save_hf_ckpt \
#     --use_wandb ${wandb_token}\
#     --wandb_run_name "Qwen2.5-3B-Instruct-En-Zh-1M-2"