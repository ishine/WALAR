#!/bin/bash

# Language Detection Script
# This script provides language detection functionality for multiple language files
# Similar to flores.sh but focused on language detection

# Configuration
MODEL_PATH="/mnt/gemini/data1/yifengliu/model/lid.176.bin"

# Language pairs configuration (similar to flores.sh)
source_language_list=(
    "eng"
    "tur"
    "ara"
    "hin"
    # "ben"
    # "guj"
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
    # "tur"
    # "uzb"
    # "kan"
    # "mal"
    # "tam"
    # "tel"
    # "est"
    # "fin"
    # "hun"
    # "kat"
    # "heb"
    # "khm"
    # "kor"
    # "tha"
)

target_language_list=(
    # 'afr' 'dan' 'nld' 'deu' 'nob' 'swe' 'cat' 'fra' 'glg' 'por' 'ron' 'spa' 'bul' 'rus' 'ita' 'ind' 'msa' 'zho_simpl' 'jpn' 'vie'
    "isl" 
    "ltz" "bel" "ces" "mkd" "pol" "slk" "slv" "ukr" "ben"
    "guj" "hin" "mar" "npi" "pan" "urd" "hye" "ell" "lav" "lit" "fas"
    "cym" "ceb" "tgl" "jav" "ara" "azj" "tur" "uzb" "kan" "mal"
    "tam" "tel" "est" "fin" "hun" "kat" "heb" "khm" "kor" "tha"

    # 'amh' 'azj' 'bel' 'isl' 'jav' 'kan' 'kor' 'kir' 'lit' 'mal'
    # 'mon' 'mar' 'mya' 'pol' 'pus' 'snd' 'som' 'srp' 'tam' 'tha'
    # 'tur' 'yor'

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
    # "tur"
    # "uzb"
    # "kan"
    # "mal"
    # "tam"
    # "tel"
    # "est"
    # "fin"
    # "hun"
    # "kat"
    # "heb"
    # "khm"
    # "kor"
    # "tha"
)

# Input directory for flores files
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B_thinking"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Llama-3.2-3B-Instruct"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/LLaMAX3-8B-Alpaca"
INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/thinking_training-Qwen3-4B-thinking_mix-1m-1M-bsz128/global_step350_hf_thinking"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores_beam/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores_beam/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"

# Convert language lists to comma-separated strings
SOURCE_LANGUAGES=$(IFS=','; echo "${source_language_list[*]}")
TARGET_LANGUAGES=$(IFS=','; echo "${target_language_list[*]}")

if [ -n "$SOURCE_LANGUAGES" ] && [ -n "$TARGET_LANGUAGES" ]; then
    # Multiple language pairs mode
    echo "Running language detection for multiple language pairs"
    echo "Source languages: $SOURCE_LANGUAGES"
    echo "Target languages: $TARGET_LANGUAGES"
    
    # Change to project directory
    cd /mnt/gemini/data1/yifengliu/qe-lr
    
    # Run the language detection for all language pairs
    python code/lang_detect.py \
        --model_path "$MODEL_PATH" \
        --input_dir "$INPUT_DIR" \
        --source_languages "$SOURCE_LANGUAGES" \
        --target_languages "$TARGET_LANGUAGES"
        
else
    echo "Error: SOURCE_LANGUAGES and TARGET_LANGUAGES must be set"
    echo "Usage examples:"
    echo "  Single language pair: LANG_PAIR=\"eng-hin\" ./lang_detect.sh"
    echo "  Multiple pairs: SOURCE_LANGUAGES=\"eng,deu\" TARGET_LANGUAGES=\"hin,ben,tam\" ./lang_detect.sh"
    exit 1
fi

