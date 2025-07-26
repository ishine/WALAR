#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
# Default values
declare -A model_path

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen2.5-3B-Instruct"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Detect-Qwen3-32B-AWQ-DA-Qwen3-4B-en-zh-1M-bsz128/global_step320_hf"
model_path["nllb"]="/mnt/gemini/data1/yifengliu/model/nllb-200-distilled-1.3B"
# zho_simpl, zho_trad, swh, tam, asm
MODEL_NAME="nllb"
MODEL_PATH=${model_path[$MODEL_NAME]}
# LANG_PAIR="eng-asm"
src="eng"
target_language_list=(
    # "ltz"
    # "mkd"
    # "pol"
    # "srp"
    # "slk"
    # "slv"
    # "ben"
    # "guj"
    # "hin"
    # "mar"
    # "ory"
    # "pan"
    # "hye"
    # "ell"
    # "lav"
    # "lit"
    # "fas"
    # "tgl"
    # "jav"
    # "ara"
    # "tur"
    # "tam"
    # "fin"

    # "tam"
    # "hin"
    # "ind"
    # "msa"
    "zho_simpl"
    # "deu"
    # "spa"
    # "rus"
    # "jpn"
    # "ara"
    # "fin"

    # "tur"
    # "asm"
    # "guj"
    # "pan"
    # "kan"
)
INPUT_DIR="/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"

if [ $MODEL_NAME == "Qwen" ]; then
    relative_path=${MODEL_PATH#*/model/}
else
    relative_path=${MODEL_PATH#*/checkpoints/}
fi

OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/${relative_path}"

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required"
fi

# Create output directory if it doesn't exist
for target in "${target_language_list[@]}"; do
    mkdir -p "$OUTPUT_DIR"
    LANG_PAIR="${src}-${target}"
    # Generate output filename
    OUTPUT_FILE="${OUTPUT_DIR}/${LANG_PAIR}.txt"

    echo "Evaluating ${LANG_PAIR} with model ${MODEL_NAME} at ${INPUT_DIR}"

    cd /mnt/gemini/data1/yifengliu/qe-lr
    # Run the evaluation
    python evaluate/flores.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_dir "$INPUT_DIR"\
        --lang_pair "$LANG_PAIR" \
        --comet22 True \
        --xcomet False \
        --tensor_parallel_size 1 \
        --output_file "$OUTPUT_FILE"
done