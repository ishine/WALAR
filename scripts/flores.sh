#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
declare -A model_path
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
model_path["llama"]="/mnt/gemini/data1/yifengliu/model/Llama-3.2-3B-Instruct"
model_path["Qwen-base"]="/mnt/gemini/data1/yifengliu/model/Qwen3-4B-Base"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/3New-Detect-New-Align-Rule-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step400_hf"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Seq-Rule-Detect-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step160_hf"
model_path["nllb"]="/mnt/gemini/data1/yifengliu/model/nllb-200-distilled-1.3B"

# Configuration
MODEL_NAME="checkpoint"
MODEL_PATH=${model_path[$MODEL_NAME]}

# Language configuration - can be set as single values or comma-separated lists
# For single language pair mode (backward compatibility)
LANG_PAIR=""  # Set this for single language pair evaluation

# For multiple language pairs mode
SOURCE_LANGUAGES="eng"  # Comma-separated list: "eng,deu,fra"
# TARGET_LANGUAGES="ben,guj,hin,mar,pan,hye,ell,lav,lit,fas,tgl,jav,ara,tur,tam,fin"  # Comma-separated list

# Legacy target_language_list for backward compatibility (will be converted to TARGET_LANGUAGES if not set)
target_language_list=(
    # "isl"
    # "ltz"
    # "bel"
    # "ces"
    # "mkd"
    # "pol"
    # "srp"
    # "slk"
    # "slv"
    # "ukr"
    # "ben"
    # "guj"
    # "hin"
    # "mar"
    # "npi"
    # "pan"
    # "urd"
    # "hye"
    # "ell"
    # "lav"
    # "lit"
    # "fas"
    # "cym"
    # "ceb"
    # "tgl"
    # "jav"
    # "ara"
    # "azj"
    # "kaz"
    # "tur"
    # "uzb"
    # "kan"
    # "mal"
    # "tam"
    # "tel"
    # "mya"
    # "est"
    # "fin"
    # "hun"
    # "kat"
    # "heb"
    # "khm"
    # "kor"
    # "lao"
    # "tha"

    "ltz"
    "mkd"
    "pol"
    "srp"
    "slk"
    "slv"
    "ben"
    "guj"
    "hin"
    "mar"
    "pan"
    "hye"
    "ell"
    "lav"
    "lit"
    "fas"
    "tgl"
    "jav"
    "ara"
    "tur"
    "tam"
    "fin"
    
    # "ben"
    # "ces"
    # "slv"
    # "ben"
    # "ell"
    # "pol"
    # "ltz"
    # "hye"
    # "azj"
    # "ces"
    # "jpn"
    # "zho_simpl"
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
    # "tam"
    # "ara"
    # "fin"

    # "tur"
    # "asm"
    # "guj"
    # "pan"
    # "kan"
)
INPUT_DIR="/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"

# Determine relative path for output directory
if [ $MODEL_NAME == "Qwen" ]; then
    relative_path=${MODEL_PATH#*/model/}
elif [ $MODEL_NAME == "Qwen-base" ]; then
    relative_path=${MODEL_PATH#*/model/}
elif [ $MODEL_NAME == "llama" ]; then
    relative_path=${MODEL_PATH#*/model/}
else
    relative_path=${MODEL_PATH#*/checkpoints/}
fi

OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/${relative_path}"


# # Convert legacy target_language_list to TARGET_LANGUAGES if TARGET_LANGUAGES is not set
# if [ -z "$TARGET_LANGUAGES" ] && [ ${#target_language_list[@]} -gt 0 ]; then
#     TARGET_LANGUAGES=$(IFS=','; echo "${target_language_list[*]}")
# fi

TARGET_LANGUAGES=$(IFS=','; echo "${target_language_list[*]}")

# Determine evaluation mode
if [ -n "$LANG_PAIR" ]; then
    # Single language pair mode (backward compatibility)
    echo "Running single language pair evaluation: $LANG_PAIR"
    
    # Generate output filename
    OUTPUT_FILE="${OUTPUT_DIR}/${LANG_PAIR}.txt"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Evaluating ${LANG_PAIR} with model ${MODEL_NAME} at ${INPUT_DIR}"
    
    cd /mnt/gemini/data1/yifengliu/qe-lr
    # Run the evaluation
    python evaluate/flores.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_dir "$INPUT_DIR" \
        --lang_pair "$LANG_PAIR" \
        --comet22 True \
        --xcomet False \
        --tensor_parallel_size $num_gpus \
        --output_file "$OUTPUT_FILE"
        
elif [ -n "$SOURCE_LANGUAGES" ] && [ -n "$TARGET_LANGUAGES" ]; then
    # Multiple language pairs mode
    echo "Running multiple language pairs evaluation"
    echo "Source languages: $SOURCE_LANGUAGES"
    echo "Target languages: $TARGET_LANGUAGES"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    cd /mnt/gemini/data1/yifengliu/qe-lr
    # Run the evaluation for all language pairs
    python evaluate/flores.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_dir "$INPUT_DIR" \
        --source_languages "$SOURCE_LANGUAGES" \
        --target_languages "$TARGET_LANGUAGES" \
        --comet22 True \
        --xcomet False \
        --tensor_parallel_size $num_gpus \
        --output_dir "$OUTPUT_DIR"
        
else
    echo "Error: Either LANG_PAIR or both SOURCE_LANGUAGES and TARGET_LANGUAGES must be set"
    echo "Usage examples:"
    echo "  Single language pair: LANG_PAIR=\"eng-hin\" ./flores.sh"
    echo "  Multiple pairs: SOURCE_LANGUAGES=\"eng,deu\" TARGET_LANGUAGES=\"hin,ben,tam\" ./flores.sh"
    exit 1
fi