eval "$(/mnt/gemini/dat1/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/data1/yifengliu/miniconda3/bin/activate qe-rl

cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

# Llama
# base_model="Qwen3-4B"
# base_model="Llama-3.2-3B-Instruct"
# LlamaX
base_model="LlamaX"
lang_detect=True
rule=True       # '\n' for metricX
masklid=False
align=False


export CUDA_VISIBLE_DEVICES=3
python -m openrlhf.cli.serve_rm \
    --model_name  metricX \
    --base_model $base_model \
    --port 2000 \
    --max_len 512 \
    --rule $rule \
    --lang_detect $lang_detect \
    --align $align \
    --masklid $masklid \
    --batch_size 16 &


# 80000
# export CUDA_VISIBLE_DEVICES=1
# python -m openrlhf.cli.serve_rm \
#     --model_name XComet\
#     --base_model $base_model \
#     --port 5555 \
#     --max_len 1536 \
#     --rule False \
#     --lang_detect False \
#     --truncate False \
#     --bleu False \
#     --align False\
#     --src $src \
#     --tgt $tgt \
#     --batch_size 16 &

# echo "XComet serves successfully!"

wait