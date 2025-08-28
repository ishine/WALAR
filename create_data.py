import json
import tqdm
from tqdm import *
import random
from transformers import AutoTokenizer


if __name__ == "__main__":
    lang_code = "ar"
    path = f"/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/{lang_code}/news.2024.{lang_code}.shuffled.deduped"
    save_path = f"/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/{lang_code}/{lang_code}1m.jsonl"
    new_dataset = []
    with open(path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # for line in lines:
    # import code; code.interact(local=locals())
    for i in tqdm(range(len(lines))):
        line = lines[i]
        tokens = tokenizer(line)
        if 10 < len(tokens['input_ids']) < 150:
            new_dataset.append({"src": line.strip(), "data_source": f"wmt24_news_crawl_{lang_code}"})
        if len(new_dataset) >= 1000000:
            break
            
    with open(save_path, 'w') as f:
        for data in new_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
