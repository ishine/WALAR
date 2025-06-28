#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# Default values
declare -A model_path

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen2.5-0.5B-Instruct"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-0.5B-en-zh-1M-bsz128/global_step120_hf"

MODEL_NAME="Qwen"
MODEL_PATH=${model_path[$MODEL_NAME]}
# zho_simpl, zho_trad, swh, tam, fra, rus
# spa(Spanish), deu(German)
LANG_PAIR="eng-deu"
INPUT_DIR="/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"
PORT=1234

if [ $MODEL_NAME == "Qwen" ]; then
    relative_path=${MODEL_PATH#*/model/}
else
    relative_path=${MODEL_PATH#*/checkpoints/}
fi
OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/${relative_path}"


server=False

if [ "$server" = True ]; then
    python3 -m sglang.launch_server --model-path ${MODEL_PATH} --host 0.0.0.0 --port ${PORT}
else
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Generate output filename
    OUTPUT_FILE="${OUTPUT_DIR}/${LANG_PAIR}.txt"

    cd /mnt/gemini/data1/yifengliu/qe-lr
    # Run the evaluation

    python evaluate/flores_sglang.py \
        --model_name_or_path "$MODEL_PATH"\
        --data_dir "$INPUT_DIR"\
        --lang_pair "$LANG_PAIR" \
        --output_file "$OUTPUT_FILE"\
        --port ${PORT}
fi