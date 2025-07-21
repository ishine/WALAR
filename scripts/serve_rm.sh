eval "$(/mnt/gemini/dat1/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/data1/yifengliu/miniconda3/bin/activate qe-rl

cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

base_model="Qwen2.5-3B-Instruct"
lang_detect=false
rule=false

# export CUDA_VISIBLE_DEVICES=2
# python -m openrlhf.cli.serve_rm \
#     --model_name  metricX\
#     --base_model $base_model \
#     --port 4000 \
#     --max_len 1536 \
#     --rule $rule \
#     --lang_detect $lang_detect \
#     --batch_size 8 &

# echo "MetricX serves successfully!"

# export CUDA_VISIBLE_DEVICES=5
# python -m openrlhf.cli.serve_rm \
#     --model_name  metricX-ref\
#     --port 4000 \
#     --max_len 1536 \
#     --src $src \
#     --tgt $tgt \
#     --lang_detect $lang_detect \
#     --batch_size 8 &

# echo "Ref MetricX serves successfully!"

# export CUDA_VISIBLE_DEVICES=3
# python -m openrlhf.cli.serve_rm \
#     --model_name Comet22\
#     --port 4000 \
#     --max_len 1536 \
#     --lang_detect False \
#     --rule False \
#     --batch_size 16 &

# echo "COMET22 serves successfully!"

# 8000
export CUDA_VISIBLE_DEVICES=4
python -m openrlhf.cli.serve_rm \
    --model_name XComet\
    --base_model $base_model \
    --port 4000 \
    --max_len 1536 \
    --batch_size 16 &

echo "XComet serves successfully!"

wait