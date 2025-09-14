#!/usr/bin/env bash
cd /mnt/gemini/data1/yifengliu/qe-lr/code

# Default values
data_name="flores"
model_name="XComet"
model_size="xl"  ### model_size can be discarded if your model_name is not XComet or metricX
dtype="bf16"  ### dtype can be discarded if your model_name is not metricX
batch_size=16 ### Should be divisible by the number of GPUs

# Language lists for batch processing
src_list=()  # Will be populated based on data_name
tgt_list=()  # Will be populated based on data_name


export CUDA_VISIBLE_DEVICES=0

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

# Function to process language pairs in batch
process_language_pairs() {
  local src_count=$1
  shift
  
  # 前 src_count 个参数是 src_list
  local src_list=("${@:1:$src_count}")
  # 后面的参数是 tgt_list
  local tgt_list=("${@:$((src_count+1))}")
  
  echo "Processing ${#src_list[@]} source languages and ${#tgt_list[@]} target languages"
  echo "Source languages: ${src_list[*]}"
  echo "Target languages: ${tgt_list[*]}"
  
  # Convert arrays to space-separated strings for Python
  local src_list_str="${src_list[*]}"
  local tgt_list_str="${tgt_list[*]}"
  
  # Determine input file pattern based on data_name
  local input_file_pattern=""
  case $data_name in
    "afriMTE")
      input_file_pattern="/mnt/gemini/data1/yifengliu/data/afriMTE/AfriMTE-ade-devtest-v2"
      ;;
    "IndicMT")
      input_file_pattern="/mnt/gemini/data1/yifengliu/data/IndicMT/collated"
      ;;
    "dev23")
      input_file_pattern="/mnt/gemini/data1/yifengliu/data/wmt23-dev/dev"
      ;;
    "test24")
      input_file_pattern="/mnt/gemini/data1/yifengliu/data/wmt24-test"
      ;;
    "low-res")
      input_file_pattern="/mnt/gemini/data1/yifengliu/data/low-res"
      ;;
    "flores")
      input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Mask+Detect-New-Align-Rule-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step500_hf"
      ;;
  esac
  
  # Run prediction with language lists
  python predict.py \
    --model_name $model_name \
    --model_size $model_size \
    --dtype $dtype \
    --max_input_length 1536 \
    --batch_size $batch_size \
    --input_file "$input_file_pattern" \
    --output_dir "/mnt/gemini/data1/yifengliu/qe-lr/output/$data_name" \
    --src_list $src_list_str \
    --tgt_list $tgt_list_str
}

### AfriMTE
if [ $data_name == "afriMTE" ]; then
  # Use provided language lists or default values
  if [ ${#src_list[@]} -eq 0 ]; then
    src_list=("ary" "eng" "yor")
  fi
  if [ ${#tgt_list[@]} -eq 0 ]; then
    tgt_list=("fra" "arz" "hau" "ibo" "kik" "luo" "som" "swh" "twi" "xho" "yor" "eng")
  fi
  
  # Use batch processing
  process_language_pairs ${#src_list[@]} "${src_list[@]}" "${tgt_list[@]}"
  
  cd /mnt/gemini/data1/yifengliu/qe-lr
  python collate_afri.py \
    --input_dir /mnt/gemini/data1/yifengliu/qe-lr/output/afriMTE/$model_name-$model_size-$dtype \

elif [ $data_name == "IndicMT" ]; then
  # Use provided language lists or default values
  if [ ${#src_list[@]} -eq 0 ]; then
    src_list=("eng")
  fi
  if [ ${#tgt_list[@]} -eq 0 ]; then
    tgt_list=("assamese" "maithili" "kannada" "punjabi")
  fi
  
  # Use batch processing
  process_language_pairs ${#src_list[@]} "${src_list[@]}" "${tgt_list[@]}"
elif [ $data_name == "dev23" ]; then
  # Use provided language lists or default values
  if [ ${#src_list[@]} -eq 0 ]; then
    src_list=("en")
  fi
  if [ ${#tgt_list[@]} -eq 0 ]; then
    tgt_list=("gu" "hi" "ta" "te")
  fi
  
  # Use batch processing
  process_language_pairs ${#src_list[@]} "${src_list[@]}" "${tgt_list[@]}"
elif [ $data_name == "test24" ]; then
  # Use provided language lists or default values
  if [ ${#src_list[@]} -eq 0 ]; then
    src_list=("en")
  fi
  if [ ${#tgt_list[@]} -eq 0 ]; then
    tgt_list=("mr")  # "yo" commented out as in original
  fi
  
  # Use batch processing
  process_language_pairs ${#src_list[@]} "${src_list[@]}" "${tgt_list[@]}"
elif [ $data_name == "low-res" ]; then
  # Use provided language lists or default values
  if [ ${#src_list[@]} -eq 0 ]; then
    src_list=("en" "es")
  fi
  if [ ${#tgt_list[@]} -eq 0 ]; then
    tgt_list=("mt" "eu")
  fi
  
  # Use batch processing
  process_language_pairs ${#src_list[@]} "${src_list[@]}" "${tgt_list[@]}"
elif [ $data_name == "flores" ]; then
  # Use provided language lists or default values
  if [ ${#src_list[@]} -eq 0 ]; then
    src_list=("eng")  # "ara" commented out as in original
  fi
  if [ ${#tgt_list[@]} -eq 0 ]; then
    tgt_list=(
      # "ben"
      # "isl" "ltz" "bel" "ces" "mkd" "pol" "srp" "slk" "slv" "ukr"
      # "guj" "hin" "mar" "npi" "pan" "urd" "hye" "ell" "lav" "lit" "fas"
      # "cym" "ceb" "tgl" "jav" "ara" "azj" "kaz" "tur" "uzb" "kan" "mal"
      # "tam" "tel" "mya" "est" "fin" "hun" "kat" "heb" "khm" "kor" "lao"
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
    )
  fi
  
  # Use batch processing
  process_language_pairs ${#src_list[@]} "${src_list[@]}" "${tgt_list[@]}"
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