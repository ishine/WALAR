eval "$(/mnt/gemini/dat1/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/data1/yifengliu/miniconda3/bin/activate qe-rl

cd /mnt/gemini/data1/yifengliu/qe-lr/openrlhf

src="en"
tgt="zh"
lang_detect=True
rule=True

export CUDA_VISIBLE_DEVICES=5
python -m openrlhf.cli.serve_rm \
    --model_name  metricX\
    --port 5000 \
    --max_len 1536 \
    --src $src \
    --tgt $tgt \
    --rule $rule \
    --lang_detect $lang_detect \
    --batch_size 8 &

echo "MetricX serves successfully!"

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

export CUDA_VISIBLE_DEVICES=3
python -m openrlhf.cli.serve_rm \
    --model_name Comet22\
    --port 4000 \
    --max_len 1536 \
    --src $src \
    --tgt $tgt \
    --lang_detect False \
    --rule False \
    --batch_size 16 &

echo "COMET22 serves successfully!"

# 8000
# export CUDA_VISIBLE_DEVICES=3
# python -m openrlhf.cli.serve_rm \
#     --model_name XComet\
#     --port 4000 \
#     --max_len 1536 \
#     --src $src \
#     --tgt $tgt \
#     --lang_detect $lang_detect \
#     --rule $rule \
#     --batch_size 16 &

# echo "XComet serves successfully!"

wait