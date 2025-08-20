#!/bin/bash
# Default values

declare -A model_path
export CUDA_VISIBLE_DEVICES=0,1

# eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
# which python
# source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

model_path["Qwen3-32B"]="/mnt/gemini/data1/yifengliu/model/Qwen3-32B"
model_path["Qwen3-32B-AWQ"]="/mnt/gemini/data1/yifengliu/model/Qwen3-32B-AWQ"
model_path["Qwen3-235B-GPTQ"]="/mnt/gemini/data1/yifengliu/model/Qwen3-235B-A22B-GPTQ-Int4"
model_path["Qwen3-235B-Instruct"]="/mnt/gemini/data1/yifengliu/model/Qwen3-235B-A22B-Instruct-2507-FP8"
model_path["Qwen3-30B-A3B"]="/mnt/gemini/data1/yifengliu/model/Qwen3-30B-A3B-Instruct-2507"

MODEL_NAME="Qwen3-30B-A3B"
data_name="IndicMT"
MAX_TOKENS=2048
EVAL_TYPE="da"
TURNS=1
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
        OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/afriMTE/${MODEL_NAME}-${EVAL_TYPE}"

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
        # "en-zh"
        "eng-assamese"
        # "eng-maithili"
        # "eng-punjabi"
        # "eng-kannada"
    )
    for pair in "${language_pairs_list[@]}"; do
        src=$(echo $pair | cut -d'-' -f1)
        tgt=$(echo $pair | cut -d'-' -f2)
        OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/IndicMT/${MODEL_NAME}-${EVAL_TYPE}"

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
            --turns ${TURNS} \
            --output_dir "$OUTPUT_DIR" 
    done
elif [ $data_name == "dev23" ]; then
  language_pairs_list=(
        "en-gu"
        "en-hi"
        "en-ta"
        "en-te"
  )

  for pair in "${language_pairs_list[@]}"; do
    src=$(echo $pair | cut -d'-' -f1)
    tgt=$(echo $pair | cut -d'-' -f2)

    OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/wmt23-dev/${MODEL_NAME}-${EVAL_TYPE}"

    mkdir -p "$OUTPUT_DIR"

    echo "Processing language pair: $src-$tgt"

    python code/qwen3.py \
        --model_name_or_path "$MODEL_PATH"\
        --input_file /mnt/gemini/data1/yifengliu/data/wmt23-dev/dev.${src}${tgt}.df.short.tsv \
        --max_tokens "$MAX_TOKENS" \
        --eval_type "$EVAL_TYPE" \
        --src "$src" \
        --tgt "$tgt" \
        --tensor_parallel_size  $num_gpus \
        --turns ${TURNS} \
        --output_dir ${OUTPUT_DIR}
  done
elif [ $data_name == "test24" ]; then
  language_pairs_list=(
      "en-yo"
      "en-mr"
  )

  for pair in "${language_pairs_list[@]}"; do
    src=$(echo $pair | cut -d'-' -f1)
    tgt=$(echo $pair | cut -d'-' -f2)

    OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/wmt24-test/${MODEL_NAME}-${EVAL_TYPE}"

    mkdir -p "$OUTPUT_DIR"

    echo "Processing language pair: $src-$tgt"

    python code/qwen3.py \
        --model_name_or_path "$MODEL_PATH"\
        --input_file /mnt/gemini/data1/yifengliu/data/wmt24-test/${src}-${tgt}.jsonl \
        --max_tokens "$MAX_TOKENS" \
        --eval_type "$EVAL_TYPE" \
        --src "$src" \
        --tgt "$tgt" \
        --tensor_parallel_size  $num_gpus \
        --turns ${TURNS} \
        --output_dir ${OUTPUT_DIR}
  done
elif [ $data_name == "low-res" ]; then
  language_pairs_list=(
      "en-mt"
      "es-eu"
  )

  for pair in "${language_pairs_list[@]}"; do
    src=$(echo $pair | cut -d'-' -f1)
    tgt=$(echo $pair | cut -d'-' -f2)

    OUTPUT_DIR="/mnt/gemini/data1/yifengliu/qe-lr/output/low-res/${MODEL_NAME}-${EVAL_TYPE}"
    mkdir -p "$OUTPUT_DIR"
    echo "Processing language pair: $src-$tgt"

    python code/qwen3.py \
        --model_name_or_path "$MODEL_PATH"\
        --input_file /mnt/gemini/data1/yifengliu/data/low-res/${src}-${tgt}.csv \
        --max_tokens "$MAX_TOKENS" \
        --eval_type "$EVAL_TYPE" \
        --src "$src" \
        --tgt "$tgt" \
        --tensor_parallel_size  $num_gpus \
        --turns ${TURNS} \
        --output_dir ${OUTPUT_DIR}
  done
fi
# 1234