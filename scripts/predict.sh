#!/usr/bin/env bash
cd /mnt/gemini/data1/yifengliu/qe-lr/code

data_name="flores"
model_name="XComet"
model_size="xl"  ### model_size can be discarded if your model_name is not XComet or metricX
dtype="bf16"  ### dtype can be discarded if your model_name is not metricX
batch_size=16 ### Should be divisible by the number of GPUs

export CUDA_VISIBLE_DEVICES=2

num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
# en->indic
#/mnt/data1/yifengliu/data/IndicMT/zero_shot/assamese.jsonl
#/mnt/data1/yifengliu/data/IndicMT/train/Guj.jsonl

# /mnt/data1/yifengliu/data/wmt-mqm-human-evaluation/generalMT2024

# AfriMTE
# /mnt/data1/yifengliu/data/afriMTE/AfriMTE-ade-devtest-v2.eng-fra.jsonl

### Support following Model Name:
### metricX
### XComet, Comet-qe-da

### AfriMTE
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

    echo "Processing language pair: $src-$tgt"

    python predict.py \
      --model_name $model_name \
      --model_size $model_size \
      --dtype $dtype \
      --max_input_length 1536 \
      --batch_size $batch_size \
      --input_file /mnt/gemini/data1/yifengliu/data/afriMTE/AfriMTE-ade-devtest-v2.$src-$tgt.jsonl \
      --output_dir /mnt/gemini/data1/yifengliu/qe-lr/output/afriMTE \
      --src $src \
      --tgt $tgt \
      --qe
  done
  cd /mnt/gemini/data1/yifengliu/qe-lr
  python collate_afri.py \
    --input_dir /mnt/gemini/data1/yifengliu/qe-lr/output/afriMTE/$model_name-$model_size-$dtype \

elif [ $data_name == "IndicMT" ]; then
  language_pairs_list=(
      "eng-assamese"
      # "eng-maithili"
      # "eng-kannada"
      # "eng-punjabi"
  )

  for pair in "${language_pairs_list[@]}"; do
    src=$(echo $pair | cut -d'-' -f1)
    tgt=$(echo $pair | cut -d'-' -f2)

    echo "Processing language pair: $src-$tgt"

    python predict.py \
      --model_name $model_name \
      --model_size $model_size \
      --dtype $dtype \
      --max_input_length 1536 \
      --batch_size ${batch_size} \
      --input_file /mnt/gemini/data1/yifengliu/data/IndicMT/collated/$tgt.jsonl \
      --output_dir /mnt/gemini/data1/yifengliu/qe-lr/output/IndicMT \
      --src $src \
      --tgt $tgt \
      --qe
  done
elif [ $data_name == "dev23" ]; then
  language_pairs_list=(
      # "en-gu"
      # "en-hi"
      # "en-ta"
      "en-te"
  )

  for pair in "${language_pairs_list[@]}"; do
    src=$(echo $pair | cut -d'-' -f1)
    tgt=$(echo $pair | cut -d'-' -f2)

    echo "Processing language pair: $src-$tgt"

    python predict.py \
      --model_name $model_name \
      --model_size $model_size \
      --dtype $dtype \
      --max_input_length 1536 \
      --batch_size ${batch_size} \
      --input_file /mnt/gemini/data1/yifengliu/data/wmt23-dev/dev.${src}${tgt}.df.short.tsv \
      --output_dir /mnt/gemini/data1/yifengliu/qe-lr/output/wmt23-dev \
      --src $src \
      --tgt $tgt \
      --qe
  done
elif [ $data_name == "test24" ]; then
  language_pairs_list=(
      "en-yo"
      "en-mr"
  )

  for pair in "${language_pairs_list[@]}"; do
    src=$(echo $pair | cut -d'-' -f1)
    tgt=$(echo $pair | cut -d'-' -f2)

    echo "Processing language pair: $src-$tgt"

    python predict.py \
      --model_name $model_name \
      --model_size $model_size \
      --dtype $dtype \
      --max_input_length 1536 \
      --batch_size ${batch_size} \
      --input_file /mnt/gemini/data1/yifengliu/data/wmt24-test/${src}-${tgt}.jsonl \
      --output_dir /mnt/gemini/data1/yifengliu/qe-lr/output/wmt24-test \
      --src $src \
      --tgt $tgt \
      --qe
  done
elif [ $data_name == "low-res" ]; then
  language_pairs_list=(
      "en-mt"
      "es-eu"
  )

  for pair in "${language_pairs_list[@]}"; do
    src=$(echo $pair | cut -d'-' -f1)
    tgt=$(echo $pair | cut -d'-' -f2)

    echo "Processing language pair: $src-$tgt"

    python predict.py \
      --model_name $model_name \
      --model_size $model_size \
      --dtype $dtype \
      --max_input_length 1536 \
      --batch_size ${batch_size} \
      --input_file /mnt/gemini/data1/yifengliu/data/low-res/${src}-${tgt}.csv \
      --output_dir /mnt/gemini/data1/yifengliu/qe-lr/output/low-res \
      --src $src \
      --tgt $tgt \
      --qe
  done
elif [ $data_name == "flores" ]; then
  src="eng"
  tgt_list=(
    # "mkd"
    # "pol"
    # "srp"
    # "slk"
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
    "tgl"
    "jav"
    "ara"
    "tur"
    "tam"
    "fin"
  )
  for tgt in "${tgt_list[@]}"; do
    # tgt="slk"
    dirname="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/New-Align-Rule-Detect-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step260_hf"
    python predict.py \
      --model_name $model_name \
      --model_size $model_size \
      --dtype $dtype \
      --max_input_length 1536 \
      --batch_size ${batch_size} \
      --input_file ${dirname}/${src}-${tgt}.txt \
      --output_dir ${dirname} \
      --src $src \
      --tgt $tgt
  done
else
  echo "Unsupported data name: $data_name"
fi
# 85.78
### IndicMT
# python predict.py \
#   --model_name metricX\
#   --max_input_length 1536 \
#   --batch_size 1 \
#   --input_file /mnt/data1/yifengliu/data/afriMTE/AfriMTE-ade-devtest-v2.yor-eng.jsonl \
#   --output_dir /mnt/data1/yifengliu/qe-lr/output/IndicMT \
#   --src yor\
#   --tgt eng\
#   --qe