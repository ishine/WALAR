import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

if __name__ == "__main__":
    tgt_lang = "ara"
    path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{tgt_lang}.devtest"
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    with open(path, 'r') as f:
        lines = f.readlines()
    length_distribution = []
    for line in lines:
        input_ids = tokenizer(line)['input_ids']
        length = len(input_ids)
        length_distribution.append(length)
    
    plt.hist(length_distribution, bins=[i for i in range(0, 200, 10)], edgecolor='black')

    plt.xlabel('Token Length')
    plt.ylabel('Count')
    plt.title('Token Length Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.show()
    plt.savefig(f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores-{tgt_lang}-token.png")
    # print(f"Token Distribution")