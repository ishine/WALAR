export CUDA_VISIBLE_DEVICES=7

cd /mnt/gemini/data1/yifengliu/qe-lr/code


python evaluate_seg.py \
  --input_file /mnt/gemini/data1/yifengliu/qe-lr/output/IndicMT/Qwen3-32B-da/eng-kannada.jsonl\
  --output_file /mnt/data1/yifengliu/qe-lr/output/temp.jsonl
