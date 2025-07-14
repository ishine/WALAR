#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Default values
declare -A model_path

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-235B-A22B-GPTQ-Int4"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Qwen2.5-0.5B-En-Zh-1M-bsz128/global_step260_hf"

# zho_simpl, zho_trad, swh, tam, asm
MODEL_NAME="Qwen"
MODEL_PATH=${model_path[$MODEL_NAME]}
# LANG_PAIR="eng-asm"
src="eng_Latn"
target_language_list=(
    # "asm"
    "mai_Deva"
    # "pan"
    # "kan"
)
INPUT_DIR="/mnt/gemini/data1/yifengliu/data/flores200_dataset/devtest"
OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores"


# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required"
fi

# Create output directory if it doesn't exist
for target in "${target_language_list[@]}"; do
    mkdir -p "$OUTPUT_DIR"
    LANG_PAIR="${src}-${target}"
    # Generate output filename
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}/flores_${LANG_PAIR}.txt"

    echo "Evaluating ${LANG_PAIR} with model ${MODEL_NAME} at ${INPUT_DIR}"

    cd /mnt/gemini/data1/yifengliu/qe-lr
    # Run the evaluation
    python evaluate/flores.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_dir "$INPUT_DIR"\
        --lang_pair "$LANG_PAIR" \
        --tensor_parallel_size 4 \
        --output_file "$OUTPUT_FILE"
done