export CUDA_VISIBLE_DEVICES=1

cd /mnt/gemini/data1/yifengliu/qe-lr/code

python evaluate_any.py \
  --input_file /mnt/gemini/data1/yifengliu/qe-lr/output/wmt23-dev/Qwen3-32B-da/en-te.jsonl \
  --output_file /mnt/gemini/data1/yifengliu/qe-lr/output/output2.jsonl