src="ar"
tgt_list=(
    "cs"
    # "hye"
    # "azj"
    # "ces"
    # "ara"
    # "tur"
    # "tam"
    # "fin"
    # "ltz"
    # "ast"
    # "oci"
    # "bos"
    # "hrv"
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
)

model_path=/mnt/gemini/data1/yifengliu/model/Qwen2.5-3B-Instruct #set your model path
template_type=base
type=openrlhf

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

for tgt in "${tgt_list[@]}"; do
    if [ ${type} == "openrlhf" ]; then
        output_file_path=/mnt/gemini/data1/yifengliu/qe-lr/data/train/${template_type}_${src}-${tgt}-1m.jsonl
    elif [ ${type} == "verl" ]; then
        output_file_path=/mnt/gemini/data1/yifengliu/qe-lr/data/train/${template_type}_${src}-${tgt}-1m.parquet
    fi
    python3 /mnt/gemini/data1/yifengliu/qe-lr/code/process_data.py \
        --src ${src} \
        --tgt ${tgt} \
        --input_file "/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/${src}/${src}1m.jsonl"\
        --tokenizer_path ${model_path} \
        --template_type ${template_type} \
        --output_file ${output_file_path} \
        --type ${type}
done