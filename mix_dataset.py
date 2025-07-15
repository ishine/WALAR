import random
import json

def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

if __name__ == "__main__":
    lang_list = ["ltz", "ast", "oci", "bos", "hrv", "mkd", "pol", "srp", "slk", "slv", "ben", "guj", "hin", "mar", "ory", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav"]
    lang_list = 
    # tenk_lang_list = ["de", "es", "ru", "ms", "ja", "zh"]
    # twenty_lang_list = ["hi", "ar", "tr", "ta"]
    num_dict = {
        "de": 10000,
        "es": 10000,
        "ru": 10000,
        "hi": 20000,
        # "ms": 10000,
        "ar": 20000,
        "tr": 20000,
        "ta": 20000,
        "ja": 10000,
        "zh": 10000,
    }
    save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-mix-mid-1m.jsonl"
    new_dataset = []
    index = 0
    for lang in lang_list:
        file_path = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-{lang}-1m.jsonl"
        dataset = load_dataset(file_path)
        new_dataset.extend(dataset[index:index + num_dict[lang]])
        index += num_dict[lang]
    
    random.shuffle(new_dataset)
    with open(save_path, 'w') as f:
        for data in new_dataset:
            f.write(json.dumps(data) + "\n")