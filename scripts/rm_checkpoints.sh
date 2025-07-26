#!/bin/bash

# Directory to clean
CHECKPOINT_DIR="/mnt/gemini/data1/yifengliu/checkpoints/Rule-Qwen3-32B-AWQ-DA-Qwen2.5-3B-Instruct-en-zh-1M-bsz128"

cd "$CHECKPOINT_DIR" || exit 1

# Loop through all matching directories
for dir in global_step*_hf; do
    # Extract the step number using parameter expansion and regex
    step_num=$(echo "$dir" | grep -oP 'global_step\K[0-9]+(?=_hf)')
    # echo $step_num
    # Check if the number is odd
    if (( step_num % 20 == 10 )); then
        echo "Removing $dir"
        rm -r "$dir"
    fi
done

