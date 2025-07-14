export CUDA_VISIBLE_DEVICES=6

cd /mnt/gemini/data1/yifengliu/qe-lr

MODEL_PATH="/mnt/gemini/data1/yifengliu/model/M-Prometheus-7B"
TURNS=4
LANG_PAIR_LIST=(
    "eng-assamese"
    # "eng-maithili"
    # "eng-kannada"
    # "eng-punjabi"
)

for pair in "${LANG_PAIR_LIST[@]}"; do
    src=$(echo $pair | cut -d'-' -f1)
    tgt=$(echo $pair | cut -d'-' -f2)
    echo "Processing language pair: $pair"
    python code/prometheus.py \
        --model_path $MODEL_PATH \
        --input_file "/mnt/gemini/data1/yifengliu/data/IndicMT/collated/$tgt.jsonl" \
        --turns $TURNS \
        --src $src \
        --tgt $tgt \
        --output_dir /mnt/gemini/data1/yifengliu/qe-lr/output/IndicMT/prometheus
done