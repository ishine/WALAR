#!/usr/bin/env bash
cd /mnt/gemini/data1/yifengliu/qe-lr/code

# Default values
# metricX, XComet
# xxl, xl
# Xcomet
# data_name="flores"
data_name="benchmax"
model_name="XComet" # XComet
model_size="xl"  ### model_size can be discarded if your model_name is not XComet or metricX
# model_name="metricX" # XComet
# model_size="xxl"  ### model_size can be discarded if your model_name is not XComet or metricX
dtype="bf16"  ### dtype can be discarded if your model_name is not metricX
batch_size=16 ### Should be divisible by the number of GPUs

# Language lists for batch processing
src_list=()  # Will be populated based on data_name
tgt_list=()  # Will be populated based on data_name

export CUDA_VISIBLE_DEVICES=7
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
      # input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128_thinking"
      input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/nllb-200-distilled-1.3B"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/schedule1024-LlamaX3-8B-schedule_mix10k-1M-bsz128/global_step1800_hf/dev"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/alpha20-LlamaX-8B-final_llamax_mix2k-bsz1024/dev"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/alpha15-LlamaX-8B-final_llamax_mix2k-bsz1024/dev"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/alpha10-LlamaX-8B--bsz1024/dev"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/final/alpha5-LlamaX-8B-final_llamax_mix2k-bsz1024/dev"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/new_qe+lang_detect-LlamaX-8B-schedule_mix10k-1M-bsz128/global_step1000_hf/dev"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B_thinking"
      ;;
      "flores_devtest")
      input_file_pattern="/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"
      ;;
      "benchmax")
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/Qwen3-4B/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule_mix-LlamaX3-8B-schedule_mix-1M-bsz128_global_step150_hf/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/global_step1250_hf/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/LLaMAX3-8B-Alpaca/flores"
      input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/pure_qe-LlamaX3-8B-schedule_mix10k-1M-bsz128_global_step1200_hf/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule1024-LlamaX3-8B-schedule_mix10k-1M-bsz128_global_step1800_hf/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/llamax2/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/Tower-Plus-9B/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/aya-expanse-8b/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_alpha10-LlamaX-8B--bsz1024/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_alpha15-LlamaX-8B-final_llamax_mix2k-bsz1024/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_alpha5-LlamaX-8B-final_llamax_mix2k-bsz1024/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/new_qe+lang_detect-LlamaX-8B-schedule_mix10k-1M-bsz128_global_step1000_hf/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_alpha20-LlamaX-8B-final_llamax_mix2k-bsz1024/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/nllb-200-1.3B-yf"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule-LlamaX3-8B-schedule-1M-bsz128_global_step800_hf/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/aya-expanse-32b/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/Tower-Plus-9B/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/hunyuan-mt/flores"
      # input_file_pattern="/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule1024-LlamaX3-8B-schedule_mix10k-1M-bsz128_global_step1800_hf/flores"
      # input_file_pattern="/mnt/gemini/data1/yatish/BenchMAX/tasks/translation/output/LLaMAX3-8B-Alpaca/flores/flores"
      ;;
  esac
  
  # Run prediction with language lists
  python predict.py \
    --model_name $model_name \
    --model_size $model_size \
    --dtype $dtype \
    --max_input_length 1024 \
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
    src_list=(
      # "eng" "rus"
      # "eng" "ara"
      # "tur" "hin"
      # "ara" "swh"
      # "rus" "zho_simpl" "swh"
      # "swh"
      # "tur" "hin"
      # "isl" "ltz" "bel" "ces" "mkd" "pol" "slk" "slv" "ukr" "ben"
      # "guj" "hin" "mar" "npi" "pan" "urd" "hye" "ell" "lav" "lit" "fas"
      # "cym" "ceb" "tgl" "jav" "ara" "azj" "tur" "uzb" "kan" "mal"
      # "tam" "tel" "est" "fin" "hun" "kat" "heb" "khm" "kor" "tha"
    
    'afr' 'amh' 'ara' 'hye' 'asm' 'ast' 'azj' 'bel' 'ben' 'bos' 'bul' 'mya' 'cat' 'ceb' 'zho_simpl' 'hrv' 'ces' 'dan' 
    'nld' 'eng' 'est' 'tgl' 'fin' 'fra' 'ful' 'glg' 'lug' 'kat' 'deu' 'ell' 'guj' 'hau' 'heb' 'hin' 'hun' 'isl' 'ibo' 
    'ind' 'gle' 'ita' 'jpn' 'jav' 'kea' 'kam' 'kan' 'kaz' 'khm' 'kor' 'kir' 'lao' 'lav' 'lin' 'lit' 'luo' 'ltz' 'mkd' 
    'msa' 'mal' 'mlt' 'mri' 'mar' 'mon' 'nob' 'npi' 'nso' 'nya' 'oci' 'ory' 'orm' 'pus' 'fas' 'pol' 'por' 'pan' 'ron' 'rus' 
    'srp' 'sna' 'snd' 'slk' 'slv' 'som' 'ckb' 'spa' 'swh' 'swe' 'tgk' 'tam' 'tel' 'tha' 'tur' 'ukr' 'umb' 'urd' 'uzb' 
    'vie' 'cym' 'wol' 'xho' 'yor' 'zul' 'zho_trad'
    )
  fi
  if [ ${#tgt_list[@]} -eq 0 ]; then
    tgt_list=(
      # "eng" "ara" "tur" "hin"
      # "rus" "zho_simpl" "swh"

      # "isl" "ltz" "bel" "ces" "mkd" "pol" "slk" "slv" "ukr" "ben"
      # "guj" "hin" "mar" "npi" "pan" "urd" "hye" "ell" "lav" "lit" "fas"
      # "cym" "ceb" "tgl" "jav" "ara" "azj" "tur" "uzb" "kan" "mal"
      # "tam" "tel" "est" "fin" "hun" "kat" "heb" "khm" "kor" "tha"
      # "isl"
      # "eng" "ara" "tur" "hin"
      # 'afr' 'dan' 'nld' 'deu' 'nob' 'swe' 'cat' 'fra' 'glg' 'por' 'ron' 'spa' 'bul' 'rus' 'ita' 'ind' 'msa' 'zho_simpl' 'jpn' 'vie'
    # "eng" "ara"
    # "tur" "hin" 
    # "rus" "zho_simpl"
    "swh"

    # 'afr' 'amh' 'ara' 'hye' 'asm' 'ast' 'azj' 'bel' 'ben' 'bos' 'bul' 'mya' 'cat' 'ceb' 'zho_simpl' 'hrv' 'ces' 'dan' 
    # 'nld' 'eng' 'est' 'tgl' 'fin' 'fra' 'ful' 'glg' 'lug' 'kat' 'deu' 'ell' 'guj' 'hau' 'heb' 'hin' 'hun' 'isl' 'ibo' 
    # 'ind' 'gle' 'ita' 'jpn' 'jav' 'kea' 'kam' 'kan' 'kaz' 'khm' 'kor' 'kir' 'lao' 'lav' 'lin' 'lit' 'luo' 'ltz' 'mkd' 
    # 'msa' 'mal' 'mlt' 'mri' 'mar' 'mon' 'nob' 'npi' 'nso' 'nya' 'oci' 'ory' 'orm' 'pus' 'fas' 'pol' 'por' 'pan' 'ron' 'rus' 
    # 'srp' 'sna' 'snd' 'slk' 'slv' 'som' 'ckb' 'spa' 'swh' 'swe' 'tgk' 'tam' 'tel' 'tha' 'tur' 'ukr' 'umb' 'urd' 'uzb' 
    # 'vie' 'cym' 'wol' 'xho' 'yor' 'zul' 'zho_trad'
    )
  fi
  
  # Use batch processing
  process_language_pairs ${#src_list[@]} "${src_list[@]}" "${tgt_list[@]}"
elif [ $data_name == "flores_devtest" ]; then
  # Use provided language lists or default values
  if [ ${#src_list[@]} -eq 0 ]; then
    src_list=(
    "mkd" "ita" "por" "ron"
    # eng,zho_simpl,rus,hin,tur,ara,swh
    )
  fi
  if [ ${#tgt_list[@]} -eq 0 ]; then
    tgt_list=(
    'afr' 'amh' 'ara' 'hye' 'asm' 'ast' 'azj' 'bel' 'ben' 'bos' 'bul' 'mya' 'cat' 'ceb' 'zho_simpl' 'hrv' 'ces' 'dan' 
    'nld' 'eng' 'est' 'tgl' 'fin' 'fra' 'ful' 'glg' 'lug' 'kat' 'deu' 'ell' 'guj' 'hau' 'heb' 'hin' 'hun' 'isl' 'ibo' 
    'ind' 'gle' 'ita' 'jpn' 'jav' 'kea' 'kam' 'kan' 'kaz' 'khm' 'kor' 'kir' 'lao' 'lav' 'lin' 'lit' 'luo' 'ltz' 'mkd' 
    'msa' 'mal' 'mlt' 'mri' 'mar' 'mon' 'nob' 'npi' 'nso' 'nya' 'oci' 'ory' 'orm' 'pus' 'fas' 'pol' 'por' 'pan' 'ron' 'rus' 
    'srp' 'sna' 'snd' 'slk' 'slv' 'som' 'ckb' 'spa' 'swh' 'swe' 'tgk' 'tam' 'tel' 'tha' 'tur' 'ukr' 'umb' 'urd' 'uzb' 
    'vie' 'cym' 'wol' 'xho' 'yor' 'zul' 'zho_trad'
    )
  fi
  
  # Use batch processing
  process_language_pairs ${#src_list[@]} "${src_list[@]}" "${tgt_list[@]}"
elif [ $data_name == "benchmax" ]; then
  # Use provided language lists or default values (ISO 2-character codes)
  if [ ${#src_list[@]} -eq 0 ]; then
    src_list=(
      # "en" "ar" "tr" "hi" "ru" "zh" "sw"
      # "en" "ar" 
      # "tr" "hi" 
      # "ru" "zh" 
      "sw"
      
      # "az" "ja" "ne"
      # 'af' 'am' 'ar' 'hy' 'as' 'ast' 'az' 'be' 'bn' 'bs' 'bg' 'my' 'ca' 'ceb' 'zh' 'zho_trad' 'hr' 'cs' 'da' 'nl' 'en' 'et' 'tl' 'fi' 'fr' 'ff' 'gl' 'lg' 'ka' 'de' 'el' 'gu' 'ha' 'he' 'hi' 'hu' 'is' 'ig' 'id' 'ga' 'it' 'ja' 'jv' 'kea' 'kam' 'kn' 'kk' 'km' 'ko' 'ky' 'lo' 'lv' 'ln' 'lt' 'luo' 'lb' 'mk' 'ms' 'ml' 'mt' 'mi' 'mr' 'mn' 'ne' 'ns' 'no' 'ny' 'oc' 'or' 'om' 'ps' 'fa' 'pl' 'pt' 'pa' 'ro' 'ru' 'sr' 'sn' 'sd' 'sk' 'sl' 'so' 'ku' 'es' 'sw' 'sv' 'tg' 'ta' 'te' 'th' 'tr' 'uk' 'umb' 'ur' 'uz' 'vi' 'cy' 'wo' 'xh' 'yo' 'zu'
      # "ja" "az" "ne"
    #  'ig' 'id' 'ga' 'it' 'ja' 'jv' 'kea' 'kam' 'kn' 'kk' 'km' 'ko' 'ky' 'lo' 'lv' 'ln' 'lt' 'luo' 'lb' 'mk' 'ms' 'ml' 'mt' 'mi' 'mr' 'mn' 'ne' 'ns' 'no' 'ny' 'oc' 'or' 'om' 'ps' 'fa' 'pl' 'pt' 'pa' 'ro' 'ru' 'sr' 'sn' 'sd' 'sk' 'sl' 'so' 'ku' 'es' 'sw' 'sv' 'tg' 'ta' 'te' 'th' 'tr' 'uk' 'umb' 'ur' 'uz' 'vi' 'cy' 'wo' 'xh' 'yo' 'zu'

    )
  fi
  if [ ${#tgt_list[@]} -eq 0 ]; then
    tgt_list=(
      # "ja" "az" "ne"
      # "en"
      # "ar" "bg" "bn" "cs" "de" "en" "es" "fi" "fr" "hi" "hu" "id" "is" "it" "mk" "nl" "pl" "pt" "ro" "ru" "sw" "tr" "uk" "zh"
      # "pt"
      'af' 'am' 'ar' 'hy' 'as' 'ast' 'az' 'be' 'bn' 'bs' 'bg' 'my' 'ca' 'ceb' 'zh' 'zho_trad' 'hr' 'cs' 'da' 'nl' 'en' 'et' 'tl' 'fi' 'fr' 'ff' 'gl' 'lg' 'ka' 'de' 'el' 'gu' 'ha' 'he' 'hi' 'hu' 'is' 'ig' 'id' 'ga' 'it' 'ja' 'jv' 'kea' 'kam' 'kn' 'kk' 'km' 'ko' 'ky' 'lo' 'lv' 'ln' 'lt' 'luo' 'lb' 'mk' 'ms' 'ml' 'mt' 'mi' 'mr' 'mn' 'ne' 'ns' 'no' 'ny' 'oc' 'or' 'om' 'ps' 'fa' 'pl' 'pt' 'pa' 'ro' 'ru' 'sr' 'sn' 'sd' 'sk' 'sl' 'so' 'ku' 'es' 'sw' 'sv' 'tg' 'ta' 'te' 'th' 'tr' 'uk' 'umb' 'ur' 'uz' 'vi' 'cy' 'wo' 'xh' 'yo' 'zu'
      # "en" "ru" "zh" "ar" "tr" "hi" "sw"
      # "en" "ar" 
      # "tr" "hi" 
      # "ru" "zh" 
      # "sw"


      # "zh" "kn" "is" "lb" "be" "cs" "mk" "sk" "sl" "uk" "bn" "gu" "hi" "mr" "ne" "pa" "ur" "hy" "el" "lv" "lt" "fa" "cy" "ceb" "tl" "jv" "ar" "az" "tr" "uz"
      # "ar" "az" "be" "bn" "ceb" "cs" "cy" "el" "fa" "gu" "hi" "hy" "is" "jv" "lb" "lt" "lv" "mk" "mr" "ne" "pa" "sk" "sl" "tl" "tr" "uk" "uz" "uz"
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
