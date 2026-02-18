#!/bin/bash

# Language Detection Script
# This script provides language detection functionality for multiple language files
# Similar to flores.sh but focused on language detection

# Configuration
MODEL_PATH="/mnt/gemini/data1/yifengliu/model/lid.176.bin"
# Toggle glotLID (FastText) support – mirrors serve_rm.py setup
USE_GLOTLID=true
GLOTLID_MODEL_PATH="/mnt/gemini/data1/yifengliu/model/models--cis-lmu--glotlid/snapshots/74cb50b709c9eefe0f790030c6c95c461b4e3b77/model.bin"
# Optional BenchMAX JSON result file (e.g., result_en-zh.json) for one-off detection
BENCHMAX_FILE=""
BENCHMAX_TARGET_LANGUAGE=""
USE_BENCHMAX_DIR=true
# USE_BENCHMAX_DIR=false

# Language pairs configuration (similar to flores.sh)
source_language_list=(
    "eng" "tur" "ara" "hin" "rus" "zho_simpl" "swh"
    # "ara"

    # 'afr' 'amh' 'ara' 'hye' 'asm' 'ast' 'azj' 'bel' 'ben' 'bos' 'bul' 'mya' 'cat' 'ceb' 'zho_simpl' 'hrv' 'ces' 'dan' 
    # 'nld' 'eng' 'est' 'tgl' 'fin' 'fra' 'ful' 'glg' 'lug' 'kat' 'deu' 'ell' 'guj' 'hau' 'heb' 'hin' 'hun' 'isl' 'ibo' 
    # 'ind' 'gle' 'ita' 'jpn' 'jav' 'kea' 'kam' 'kan' 'kaz' 'khm' 'kor' 'kir' 'lao' 'lav' 'lin' 'lit' 'luo' 'ltz' 'mkd' 
    # 'msa' 'mal' 'mlt' 'mri' 'mar' 'mon' 'nob' 'npi' 'nso' 'nya' 'oci' 'ory' 'orm' 'pus' 'fas' 'pol' 'por' 'pan' 'ron' 'rus' 
    # 'srp' 'sna' 'snd' 'slk' 'slv' 'som' 'ckb' 'spa' 'swh' 'swe' 'tgk' 'tam' 'tel' 'tha' 'tur' 'ukr' 'umb' 'urd' 'uzb' 
    # 'vie' 'cym' 'wol' 'xho' 'yor' 'zul' 'zho_trad'
)

target_language_list=(
    # 'afr' 'dan' 'nld' 'deu' 'nob' 'swe' 'cat' 'fra' 'glg' 'por' 'ron' 'spa' 'bul' 'rus' 'ita' 'ind' 'msa' 'zho_simpl' 'jpn' 'vie'
    # "som" 
    # "ltz" "bel" "ces" "mkd" "pol" "slk" "slv" "ukr" "ben"
    # "guj" "hin" "mar" "npi" "pan" "urd" "hye" "ell" "lav" "lit" "fas"
    # "cym" "ceb" "tgl" "jav" "ara" "azj" "tur" "uzb" "kan" "mal"
    # "tam" "tel" "est" "fin" "hun" "kat" "heb" "khm" "kor" "tha"

    # 'amh' 'azj' 'bel' 'isl' 'jav' 'kan' 'kor' 'kir' 'lit' 'mal'
    # 'mon' 'mar' 'mya' 'pol' 'pus' 'snd' 'som' 'srp' 'tam' 'tha'
    # 'tur' 'yor'

    # "eng" "tur" "ara" "hin" "rus" "zho_simpl" "swh"

    'afr' 'amh' 'ara' 'hye' 'asm' 'ast' 'azj' 'bel' 'ben' 'bos' 'bul' 'mya' 'cat' 'ceb' 'zho_simpl' 'hrv' 'ces' 'dan' 
    'nld' 'eng' 'est' 'tgl' 'fin' 'fra' 'ful' 'glg' 'lug' 'kat' 'deu' 'ell' 'guj' 'hau' 'heb' 'hin' 'hun' 'isl' 'ibo' 
    'ind' 'gle' 'ita' 'jpn' 'jav' 'kea' 'kam' 'kan' 'kaz' 'khm' 'kor' 'kir' 'lao' 'lav' 'lin' 'lit' 'luo' 'ltz' 'mkd' 
    'msa' 'mal' 'mlt' 'mri' 'mar' 'mon' 'nob' 'npi' 'nso' 'nya' 'oci' 'ory' 'orm' 'pus' 'fas' 'pol' 'por' 'pan' 'ron' 'rus' 
    'srp' 'sna' 'snd' 'slk' 'slv' 'som' 'ckb' 'spa' 'swh' 'swe' 'tgk' 'tam' 'tel' 'tha' 'tur' 'ukr' 'umb' 'urd' 'uzb' 
    'vie' 'cym' 'wol' 'xho' 'yor' 'zul' 'zho_trad'
    # "ara"
    # "zho_simpl"
    # "amh"
)

# Input directory for flores files
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B_thinking"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Llama-3.2-3B-Instruct"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/LLaMAX3-8B-Alpaca"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/schedule1024-LlamaX3-8B-schedule_mix10k-1M-bsz128/global_step1800_hf/dev"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/alpha20-LlamaX-8B-final_llamax_mix2k-bsz1024/dev"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/alpha15-LlamaX-8B-final_llamax_mix2k-bsz1024/dev"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/alpha10-LlamaX-8B--bsz1024/dev"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/alpha5-LlamaX-8B-final_llamax_mix2k-bsz1024/dev"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/new_qe+lang_detect-LlamaX-8B-schedule_mix10k-1M-bsz128/global_step1000_hf/dev"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/nllb-200-distilled-1.3B"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/nllb-200-1.3B-yf"
INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/pure_qe-LlamaX3-8B-schedule_mix10k-1M-bsz128_global_step1200_hf/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/nllb-200-1.3B"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/LLaMAX3-8B-Alpaca/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/nllb-200-1.3B-yf"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/hunyuan-mt/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/Tower-Plus-9B/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/aya-expanse-8b/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/pure_qe-LlamaX3-8B-schedule_mix10k-1M-bsz128_global_step1200_hf/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_alpha10-LlamaX-8B--bsz1024/flores"

# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/aya-expanse-8b/flores"
# INPUT_DIR=""/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/Tower-Plus-9B/flores""
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/thinking_training-Qwen3-4B-thinking_mix-1m-1M-bsz128/global_step350_hf_thinking"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/LLaMAX3-8B-Alpaca/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/new_qe+lang_detect-LlamaX-8B-schedule_mix10k-1M-bsz128_global_step1000_hf/flores"
# INPUT_DIR=""/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_alpha5-LlamaX-8B-final_llamax_mix2k-bsz1024/flores""
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_alpha10-LlamaX-8B--bsz1024/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_alpha15-LlamaX-8B-final_llamax_mix2k-bsz1024/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_alpha20-LlamaX-8B-final_llamax_mix2k-bsz1024/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule1024-LlamaX3-8B-schedule_mix10k-1M-bsz128_global_step1800_hf/flores"
# INPUT_DIR="/mnt/gemini/data1/yatish/BenchMAX/tasks/translation/output/LLaMAX3-8B-Alpaca/flores/flores"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores_beam/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"
# INPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/flores_beam/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"

# Convert language lists to comma-separated strings
SOURCE_LANGUAGES=$(IFS=','; echo "${source_language_list[*]}")
TARGET_LANGUAGES=$(IFS=','; echo "${target_language_list[*]}")


if [ "$USE_BENCHMAX_DIR" = "true" ]; then
    echo "Running BenchMAX directory mode against $INPUT_DIR"
    cd /mnt/gemini/data1/yifengliu/qe-lr || exit 1

    EXTRA_ARGS=()
    if [ "$USE_GLOTLID" = "true" ]; then
        EXTRA_ARGS+=(--use_glotlid)
        EXTRA_ARGS+=(--glotlid_model_path "$GLOTLID_MODEL_PATH")
    fi

    python code/lang_detect.py \
        --model_path "$MODEL_PATH" \
        --input_dir "$INPUT_DIR" \
        --source_languages "$SOURCE_LANGUAGES" \
        --target_languages "$TARGET_LANGUAGES" \
        --benchmax \
        "${EXTRA_ARGS[@]}"

    exit 0
fi

if [ -n "$SOURCE_LANGUAGES" ] && [ -n "$TARGET_LANGUAGES" ]; then
    # Multiple language pairs mode
    echo "Running language detection for multiple language pairs"
    echo "Source languages: $SOURCE_LANGUAGES"
    echo "Target languages: $TARGET_LANGUAGES"
    
    # Change to project directory
    cd /mnt/gemini/data1/yifengliu/qe-lr
    
    # Optional glotLID arguments
    EXTRA_ARGS=()
    if [ "$USE_GLOTLID" = "true" ]; then
        EXTRA_ARGS+=(--use_glotlid)
        EXTRA_ARGS+=(--glotlid_model_path "$GLOTLID_MODEL_PATH")
    fi

    # Run the language detection for all language pairs
    python code/lang_detect.py \
        --model_path "$MODEL_PATH" \
        --input_dir "$INPUT_DIR" \
        --source_languages "$SOURCE_LANGUAGES" \
        --target_languages "$TARGET_LANGUAGES" \
        "${EXTRA_ARGS[@]}"
        
else
    echo "Error: SOURCE_LANGUAGES and TARGET_LANGUAGES must be set"
    echo "Usage examples:"
    echo "  Single language pair: LANG_PAIR=\"eng-hin\" ./lang_detect.sh"
    echo "  Multiple pairs: SOURCE_LANGUAGES=\"eng,deu\" TARGET_LANGUAGES=\"hin,ben,tam\" ./lang_detect.sh"
    exit 1
fi

