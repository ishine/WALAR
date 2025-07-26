import json
import copy
import os
from tqdm import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_dataset(path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def my_load_dataset(data_pair, lang):
    dataset = []
    path = os.path.join(data_pair, f"{lang}.devtest")
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(line.strip())
    return dataset

def load_flores_dataset(data_dir, lang_pair):
    """Load FLORES-101 dataset for a specific language pair."""
    # dataset = load_dataset("facebook/flores", "all")
    src_lang, tgt_lang = lang_pair.split("-")
    
    # Get test split
    src_dataset, tgt_dataset = my_load_dataset(data_dir, src_lang), my_load_dataset(data_dir, tgt_lang)
    
    # Filter for the specific language pair
    return src_dataset, tgt_dataset
        
    
if __name__ == '__main__':
    src_lang = "eng_Latn"
    tgt_lang = "deu_Latn"
    model_path = "/mnt/gemini/data1/yifengliu/model/nllb-200-distilled-1.3B"
    path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-zh-1m.jsonl"
    save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-zh2-1m.jsonl"
    dataset = load_dataset(path)
    # dataset = my_load_dataset("/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest", "eng")
    new_dataset = []
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=True, src_lang=src_lang
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=True)
    model.to("cuda:0")
    batch_size = 1
    # dataset = dataset[:10]
    for i in range(0, len(dataset), batch_size):
        right_bound = min(i + batch_size, len(dataset))
        data = dataset[i:right_bound]
        new_data = [copy.deepcopy(dt) for dt in data]
        sources = [dt['src'] for dt in data]
        # chunks = src.split('.')[:-1] if '.' in src else [src]
        # src = src[:-1]
        inputs = tokenizer(sources, return_tensors="pt", padding=True)
        inputs.to("cuda:0")
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang), max_length=512
        )
        output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True, model_max_length=512)
        new_data['ref'] = output
        new_dataset.append(new_data)
    import code; code.interact(local=locals())
    # with open(save_path, 'w') as f:
    #     for data in new_dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + '\n')