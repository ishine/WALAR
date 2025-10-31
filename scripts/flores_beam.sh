#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
declare -A model_path
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-8B"
model_path["Qwen3X"]="/mnt/gemini/data1/yifengliu/model/Qwen3-XPlus-8B"
model_path["llama"]="/mnt/gemini/data1/yifengliu/model/Llama-3.2-3B-Instruct"
model_path["llamax"]="/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca"
model_path["Qwen-base"]="/mnt/gemini/data1/yifengliu/model/Qwen3-4B-Base"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Continue-Mask+Detect-New-Align-Rule-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step700_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Final-mix-LlamaX3-8B-final_llamax_mix-1m-1M-bsz128/global_step300_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule-LlamaX3-8B-schedule-1M-bsz128/global_step400_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Final-Llama3.2-3B-final_mix-160k-1M-bsz128/global_step480_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/new_eight_directions-LlamaX3-4B-new_eight_directions-llamax-mix-1m-1M-bsz128/global_step400_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128/global_step50_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128"
# model_path["generalization"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Generalization-Qwen3-4B-final_en-mix-1m-1M-bsz128"
model_path["nllb"]="/mnt/gemini/data1/yifengliu/model/nllb-200-distilled-1.3B"

# Configuration
MODEL_NAME="checkpoint"
MODEL_PATH=${model_path[$MODEL_NAME]}

# For multiple language pairs mode
# SOURCE_LANGUAGES="ara"  # Comma-separated list: "eng,deu,fra"
# TARGET_LANGUAGES="ben,guj,hin,mar,pan,hye,ell,lav,lit,fas,tgl,jav,ara,tur,tam,fin"  # Comma-separated list
source_language_list=(
    "eng"
    # "hin"
    # "ara"
    # "tur"
    # "hin"
    # "isl" "ltz" "bel" "ces" "mkd" "pol" "slk" "slv" "ukr" "ben"
    # "guj" "hin" "mar" "npi" "pan" "urd" "hye" "ell" "lav" "lit" "fas"
    # "cym" "ceb" "tgl" "jav" "ara" "azj" "tur" "uzb" "kan" "mal"
    # "tam" "tel" "est" "fin" "hun" "kat" "heb" "khm" "kor" "tha"

    # 'amh' 'azj' 'bel' 'isl' 'jav'
    # 'kan' 'kor' 'kir' 'lit' 'mal'
    # 'mon' 'mar' 'mya' 'pol' 'pus'
    # 'snd' 'som' 'srp' 'tam' 'tha'
    # 'tur' 'yor'

    # 'afr' 'dan' 'nld' 'deu' 'nob' 'swe' 'cat' 'fra' 'glg' 'por' 'ron' 'spa' 'bul' 'rus' 'ita' 'ind' 'msa' 'zho_simpl' 'jpn' 'vie'
)
# Legacy target_language_list for backward compatibility (will be converted to TARGET_LANGUAGES if not set)
target_language_list=(
    # "eng"
    # "ara"
    # "tur"
    # "hin"
    "isl" "ltz" "bel" "ces" "mkd" "pol" "slk" "slv" "ukr" "ben"
    "guj" "hin" "mar" "npi" "pan" "urd" "hye" "ell" "lav" "lit" "fas"
    "cym" "ceb" "tgl" "jav" "ara" "azj" "tur" "uzb" "kan" "mal"
    "tam" "tel" "est" "fin" "hun" "kat" "heb" "khm" "kor" "tha"

    # 'afr' 'amh' 'ara' 'hye' 'asm' 'ast' 'azj' 'bel' 'ben' 'bos' 'bul' 'mya' 'cat' 'ceb' 'zho_simpl' 'hrv' 'ces' 'dan' 
    # 'nld' 'eng' 'est' 'tgl' 'fin' 'fra' 'ful' 'glg' 'lug' 'kat' 'deu' 'ell' 'guj' 'hau' 'heb' 'hin' 'hun' 'isl' 'ibo' 
    # 'ind' 'gle' 'ita' 'jpn' 'jav' 'kea' 'kam' 'kan' 'kaz' 'khm' 'kor' 'kir' 'lao' 'lav' 'lin' 'lit' 'luo' 'ltz' 'mkd' 
    # 'msa' 'mal' 'mlt' 'mri' 'mar' 'mon' 'nob' 'npi' 'nso' 'nya' 'oci' 'ory' 'orm' 'pus' 'fas' 'pol' 'por' 'pan' 'ron' 'rus' 
    # 'srp' 'sna' 'snd' 'slk' 'slv' 'som' 'ckb' 'spa' 'swh' 'swe' 'tgk' 'tam' 'tel' 'tha' 'tur' 'ukr' 'umb' 'urd' 'uzb' 
    # 'vie' 'cym' 'wol' 'xho' 'yor' 'zul' 'zho_trad'
)
INPUT_DIR="/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"

# Determine relative path for output directory
if [ $MODEL_NAME == "Qwen" ]; then
    relative_path=${MODEL_PATH#*/model/}
elif [ $MODEL_NAME == "Qwen-base" ]; then
    relative_path=${MODEL_PATH#*/model/}
elif [ $MODEL_NAME == "Qwen3X" ]; then
    relative_path=${MODEL_PATH#*/model/}
elif [ $MODEL_NAME == "llama" ]; then
    relative_path=${MODEL_PATH#*/model/}
elif [ $MODEL_NAME == "llamax" ]; then
    relative_path=${MODEL_PATH#*/model/}
else
    relative_path=${MODEL_PATH#*/checkpoints/}
fi

OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores_beam/${relative_path}"


# # Convert legacy target_language_list to TARGET_LANGUAGES if TARGET_LANGUAGES is not set
# if [ -z "$TARGET_LANGUAGES" ] && [ ${#target_language_list[@]} -gt 0 ]; then
#     TARGET_LANGUAGES=$(IFS=','; echo "${target_language_list[*]}")
# fi
SOURCE_LANGUAGES=$(IFS=','; echo "${source_language_list[*]}")
TARGET_LANGUAGES=$(IFS=','; echo "${target_language_list[*]}")


if [ -n "$SOURCE_LANGUAGES" ] && [ -n "$TARGET_LANGUAGES" ]; then
    # Multiple language pairs mode
    echo "Running multiple language pairs evaluation"
    echo "Source languages: $SOURCE_LANGUAGES"
    echo "Target languages: $TARGET_LANGUAGES"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    cd /mnt/gemini/data1/yifengliu/qe-lr
    # Run the evaluation for all language pairs
    python evaluate/flores_beam.py \
        --model_name_or_path "$MODEL_PATH" \
        --data_dir "$INPUT_DIR" \
        --source_languages "$SOURCE_LANGUAGES" \
        --target_languages "$TARGET_LANGUAGES" \
        --comet22 False \
        --xcomet False \
        --batch_size 8 \
        --num_beams 4 \
        --max_new_tokens 256 \
        --length_penalty 1.0 \
        --no_repeat_ngram_size 3 \
        --output_dir "$OUTPUT_DIR"
        
else
    echo "Error: Either LANG_PAIR or both SOURCE_LANGUAGES and TARGET_LANGUAGES must be set"
    echo "Usage examples:"
    echo "  Single language pair: LANG_PAIR=\"eng-hin\" ./flores_beam.sh"
    echo "  Multiple pairs: SOURCE_LANGUAGES=\"eng,deu\" TARGET_LANGUAGES=\"hin,ben,tam\" ./flores_beam.sh"
    exit 1
fi
