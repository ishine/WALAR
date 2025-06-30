src="en"
tgt="ru" # all you need to care

model_path=/mnt/gemini/data1/yifengliu/model/Qwen2.5-0.5B-Instruct #set your model path
template_type=base
type=openrlhf
if [ ${type} == "openrlhf" ]; then
    output_file_path=/mnt/gemini/data1/yifengliu/qe-lr/data/train/${template_type}_${src}-${tgt}-1m.jsonl
elif [ ${type} == "verl" ]; then
    output_file_path=/mnt/gemini/data1/yifengliu/qe-lr/data/train/${template_type}_${src}-${tgt}-1m.parquet
fi

eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl


python3 /mnt/gemini/data1/yifengliu/qe-lr/code/process_data.py \
    --src ${src} \
    --tgt ${tgt} \
    --input_file "/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/en/en1m.jsonl"\
    --tokenizer_path ${model_path} \
    --template_type ${template_type} \
    --output_file ${output_file_path} \
    --type ${type}