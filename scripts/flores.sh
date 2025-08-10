#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
declare -A model_path
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-30B-A3B-Instruct-2507"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Detect-Back-Translation-MetricX-Bleu-Qwen2.5-3B-Instruct-en-zh-1M-bsz128/global_step70_hf"
model_path["nllb"]="/mnt/gemini/data1/yifengliu/model/nllb-200-distilled-1.3B"
# /mnt/gemini/data1/yifengliu/checkpoints/Back-Translation-0.06-Qwen3-4B-en-zh-1M-bsz128/global_step120_hf
# zho_simpl, zho_trad, swh, tam, asm
MODEL_NAME="checkpoint"
MODEL_PATH=${model_path[$MODEL_NAME]}
# LANG_PAIR="eng-asm"
# src="zho_simpl"
src="eng"
# src="deu"
target_language_list=(
    # "eng"
    "zho_simpl"
    # "srp"
    # "afr"
    # "dan"
    # "nld"
    # "deu"
    # "isl"
    # "nob"
    # "swe"
    # "ast"
    # "cat"
    # "fra"
    # "glg"
    # "oci"
    # "por"
    # "ron"
    # "spa"
    # "bel"
    # "bos"
    # "bul"
    # "hrv"
    # "ces"
    # "rus"
    # "ukr"
    # "asm"
    # "npi"
    # "ory"
    # "pan"
    # "snd"
    # "urd"
    
    # "gle"
    # "cym"
    # "ita"
    # "pus"
    # "ckb"
    # "tgk"
    # "ceb"
    # "ind"
    # "msa"
    # "mri"
    # "lug"
    # "ibo"
    # "kea"
    # "kam"
    # "lin"
    # "nso"
    # "nya"
    # "sna"
    # "swh"
    # "umb"
    # "wol"
    # "xho"
    # "yor"
    # "zul"
    # "amh"
    # "ful"
    # "mlt"
    # "som"
    # "azj"
    # "kaz"
    # "kir"
    # "uzb"
    # "kan"
    # "mal"
    # "tel"
    # "mya"
    # "est"
    # "hun"
    # "kat"
    # "hau"
    # "heb"
    # "jpn"
    # "khm"
    # "vie"
    # "kor"
    # "lao"
    # "tha"
    # "luo"
    # "mon"

    # "eng"
    # "uzb"
    # "tel"
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
    # "orm"
    # "swh"
    # "tam"
    # "hin"
    # "ind"
    # "msa"
    # "zho_simpl"
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
        --tensor_parallel_size $num_gpus \
        --output_file "$OUTPUT_FILE"
done