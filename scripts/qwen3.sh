#!/bin/bash
# Default values

declare -A model_path
export CUDA_VISIBLE_DEVICES=4,5,6,7
eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-32B"

# "/mnt/gemini/data1/yifengliu/model/Qwen3-235B-A22B-GPTQ-Int4"

# /mnt/gemini/data1/yifengliu/checkpoints/Qwen2.5-0.5B-En-Zh-1M-bsz128/global_step140_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-zh-1M-bsz128/global_step120_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-ru-1M-bsz128/global_step120_hf
# /mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-3B-en-zh-1M-bsz128/global_step120_hf

MODEL_NAME="Qwen"
data_name="IndicMT"
MAX_TOKENS=2048
EVAL_TYPE="da"
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
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

cd /mnt/gemini/data1/yifengliu/qe-lr


if [ $data_name == "afriMTE" ]; then
    language_pairs_list=(
        "ary-fra"
        "eng-arz"
        "eng-fra"
        "eng-hau"
        "eng-ibo"
        "eng-kik"
        "eng-luo"
        "eng-som"
        "eng-swh"
        "eng-twi"
        "eng-xho"
        "eng-yor"
        "yor-eng"
    )
    for pair in "${language_pairs_list[@]}"; do
        src=$(echo $pair | cut -d'-' -f1)
        tgt=$(echo $pair | cut -d'-' -f2)
        OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/afriMTE/Qwen3-32B-${EVAL_TYPE}"

        mkdir -p "$OUTPUT_DIR"
        INPUT_FILE="/mnt/gemini/data1/yifengliu/data/afriMTE/AfriMTE-ade-devtest-v2.${src}-${tgt}2.jsonl"
        echo "Processing language pair: $src-$tgt"

        python code/qwen3.py \
            --model_name_or_path "$MODEL_PATH"\
            --input_file "$INPUT_FILE" \
            --max_tokens "$MAX_TOKENS" \
            --eval_type "$EVAL_TYPE" \
            --src "$src" \
            --tgt "$tgt" \
            --tensor_parallel_size  $num_gpus \
            --output_dir "$OUTPUT_DIR" 

        python collate_afri.py \
            --input_dir $OUTPUT_DIR 
    done
elif [ $data_name == "IndicMT" ]; then
    language_pairs_list=(
        "eng-assamese"
        "eng-maithili"
        "eng-punjabi"
        "eng-kannada"
    )
    for pair in "${language_pairs_list[@]}"; do
        src=$(echo $pair | cut -d'-' -f1)
        tgt=$(echo $pair | cut -d'-' -f2)
        OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/IndicMT/Qwen3-32B-${EVAL_TYPE}"

        mkdir -p "$OUTPUT_DIR"
        INPUT_FILE="/mnt/gemini/data1/yifengliu/data/IndicMT/collated/${tgt}2.jsonl"
        echo "Processing language pair: $src-$tgt"

        python code/qwen3.py \
            --model_name_or_path "$MODEL_PATH"\
            --input_file "$INPUT_FILE" \
            --max_tokens "$MAX_TOKENS" \
            --eval_type "$EVAL_TYPE" \
            --src "$src" \
            --tgt "$tgt" \
            --tensor_parallel_size  $num_gpus \
            --output_dir "$OUTPUT_DIR" 

    done
fi
# 1234


    # Generate output filename

# Run the evaluation

# echo "evaluating ${LANG_PAIR} with model ${MODEL_NAME} at ${INPUT_FILE}"
# python code/qwen3.py \
#     --model_name_or_path "$MODEL_PATH"\
#     --input_file "$INPUT_FILE" \
#     --max_tokens "$MAX_TOKENS" \
#     --eval_type "$EVAL_TYPE" \
#     --tensor_parallel_size  $num_gpus \
#     --output_dir "$OUTPUT_DIR" \
