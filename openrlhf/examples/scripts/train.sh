export CUDA_VISIBLE_DEVICES=4,5,6,7
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${num_gpus}


CONDA_PATH=/mnt/gemini/data1/yifengliu/miniconda3
eval "$(${CONDA_PATH}/bin/conda shell.bash hook)"
which python
source ${CONDA_PATH}/bin/activate qe-rl

OPENRLHF_PATH=/mnt/gemini/data1/yifengliu/qe-lr/openrlhf


cd $OPENRLHF_PATH

export HF_HOME=/mnt/gemini/data2/yifengliu/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/datasets
export HF_HUB_CACHE=/mnt/gemini/data2/yifengliu/.cache/huggingface/hub
export DS_SKIP_CUDA_CHECK=1
export RAY_DEBUG_POST_MORTEM=1


declare -A path_dict
path_dict["Llama"]="/mnt/gemini/data1/yifengliu/model/Llama-3.2-3B-Instruct"
path_dict["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-8B"
path_dict["glotmax"]="/mnt/gemini/data1/yifengliu/model/GlotMAX-101-8B"
path_dict["LlamaX"]="/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca"
path_dict["gemma"]="/mnt/gemini/data1/yifengliu/model/translategemma-4b-it"
path_dict["aya"]="/mnt/gemini/data1/yifengliu/model/aya-expanse-8b"

# model="gemma"
# dataname="final_gemma_mix250"
# size="8B"
# reward_name="temp2"

model="Qwen"
dataname="final_qwen_mix250"
size="8B"
reward_name="alpha20"


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
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_sync_backend nccl \
    --ref_reward_offload \
    --pretrain ${path_dict[$model]} \
    --remote_rm_url http://localhost:2000/get_reward \
    --micro_train_batch_size 16 \
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