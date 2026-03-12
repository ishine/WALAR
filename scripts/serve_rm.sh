export CUDA_VISIBLE_DEVICES=1

CONDA_PATH=/mnt/gemini/data1/yifengliu/miniconda3
OPENRLHF_PATH=/mnt/gemini/data1/yifengliu/qe-lr/openrlhf

eval "$(${CONDA_PATH}/bin/conda shell.bash hook)"
which python
source $CONDA_PATH/bin/activate qe-rl

cd $OPENRLHF_PATH


# LlamaX
# Gemma
base_model="Qwen"
lang_detect=True
masklid=True
align=True
alpha=20

python -m openrlhf.cli.serve_rm \
    --model_name  metricX \
    --base_model $base_model \
    --port 2000 \
    --max_len 512 \
    --lang_detect $lang_detect \
    --align $align \
    --masklid $masklid \
    --alpha $alpha \
    --batch_size 16 &


# You can also run xcomet
# export CUDA_VISIBLE_DEVICES=1
# python -m openrlhf.cli.serve_rm \
#     --model_name XComet\
#     --base_model $base_model \
#     --port 2000 \
#     --max_len 512 \
#     --lang_detect $lang_detect \
#     --align $align\
#     --alpha $alpha \
#     --batch_size 16 &


wait