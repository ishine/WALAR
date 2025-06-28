#!/usr/bin/env bash
cd /mnt/gemini/data1/yifengliu/qe-lr/code

data_name="IndicMT"
model_name="XComet"
model_size="xl"  ### model_size can be discarded if your model_name is not XComet or metricX
dtype="bf16"  ### dtype can be discarded if your model_name is not metricX
batch_size=16 ### Should be divisible by the number of GPUs

# export CUDA_VISIBLE_DEVICES=0

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
else
  echo "Unsupported data name: $data_name"
fi

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