export CUDA_VISIBLE_DEVICES=7

cd /mnt/gemini/data1/yifengliu/qe-lr/code


python evaluate_seg.py \
  --input_file /mnt/gemini/data1/yifengliu/qe-lr/output/low-res/metricX-xxl-bf16/es-eu.jsonl \
  --output_file /mnt/data1/yifengliu/qe-lr/output/temp.jsonl
