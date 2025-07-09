#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
# Default values
declare -A model_path

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen3-4B-en-mix-1M-bsz128/global_step120_hf"

# /mnt/gemini/data1/yifengliu/checkpoints/Qwen2.5-0.5B-En-Zh-1M-bsz128/global_step140_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-zh-1M-bsz128/global_step120_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-ru-1M-bsz128/global_step120_hf

MODEL_NAME="Qwen"
MODEL_PATH=${model_path[$MODEL_NAME]}
# zho_simpl, zho_trad, swh, tam, fra, rus
# spa(Spanish), deu(German)， heb(Hebrew)
# ben(Bengali), hin(Hindi)
# jpn(Japanese)
# tgl(fillipino Tagalog)
# fin(Finnish)
# ara(Arabic)
# tur(Turkish)
# LANG_PAIR="zho_simpl-deu"
INPUT_DIR="/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"
PORT=1234
MAX_TOKENS=512
source_language="eng"
target_language_list=(
    # "zho_simpl"
    # "fra"
    # "deu"
    # "jpn"
    # "spa"
    # "rus"
    # "fin"
    "ara"
    # "tur"
    # "ben"
    # "hin"
    # "swh"
    # "tam"
    # "bel"
    # "pol"
    # "ukr"
    # "kea"
    # "nso"
    # "ind"
    # "msa"
    # "mlt"
    # "mkd"
    # "slk"
    # "glg"
    # "oci"
)
server=False
# 1234

if [ $MODEL_NAME == "Qwen" ]; then
    relative_path=${MODEL_PATH#*/model/}
else
    relative_path=${MODEL_PATH#*/checkpoints/}
fi
OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/${relative_path}"


if [ "$server" = True ]; then
    # python3 -m sglang.launch_server --model-path ${MODEL_PATH} --host 0.0.0.0 --port ${PORT}
    python -m sglang.launch_server --model-path ${MODEL_PATH} --host 0.0.0.0 --port ${PORT} --chat-template /mnt/gemini/data1/yifengliu/qe-lr/config/qwen3_nonthinking.jinja
else
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    for target in "${target_language_list[@]}"; do
        # Generate output filename
        LANG_PAIR="${source_language}-${target}"
        OUTPUT_FILE="${OUTPUT_DIR}/${LANG_PAIR}.txt"

        cd /mnt/gemini/data1/yifengliu/qe-lr
        # Run the evaluation
        echo "evaluating ${LANG_PAIR} with model ${MODEL_NAME} at ${OUTPUT_FILE}"
        python evaluate/flores_sglang.py \
            --model_name_or_path "$MODEL_PATH"\
            --data_dir "$INPUT_DIR"\
            --lang_pair "$LANG_PAIR" \
            --max_tokens "$MAX_TOKENS" \
            --comet22 False \
            --xcomet False \
            --output_file "$OUTPUT_FILE"\
            --port ${PORT}
    done
fi