export CUDA_VISIBLE_DEVICES=1

cd /mnt/gemini/data1/yifengliu/qe-lr/code

# Qwen3-235B-da
# metricX-xxl-bf16
python evaluate_seg.py \
  --input_file /mnt/gemini/data1/yifengliu/qe-lr/output/dev23/metricX-xxl-bf16-align/en-gu.jsonl \
  --output_file /mnt/data1/yifengliu/qe-lr/output/temp.jsonl
