import json
import tqdm
from tqdm import *
import random
import spacy
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from transformers import pipeline

# def get_length_distribution(length_distribution, binsize=10):
#     """统计数据集的长度分布（按桶）"""
#     # lengths = [len(s.split()) for s, _ in dataset]
#     # token_lengths = [len(tokenizer(s)['input_ids']) for s in dataset]
#     buckets = [l // binsize for l in length_distribution]  # 例如长度 1-10 -> 0 桶
#     counter = Counter(buckets)
#     # total = sum(counter.values())
#     total = len(length_distribution)
#     dist = {k: v / total for k, v in counter.items()}
#     return dist

# def sample_trainset(length_distribution, test_dist, binsize=10):
#     """根据测试集的长度分布，从训练集中采样"""
#     # 训练集按桶分类
#     train_buckets = defaultdict(list)
#     for length in length_distribution:
#         # b = (len(s.split()) - 1) // binsize
#         # token_length = len(tokenizer(data)['input_ids'])
#         train_buckets[length//binsize].append(data)

#     # 总训练样本数（按比例决定）
#     total_samples = 120_0000
    
#     new_train = []
#     for b, ratio in test_dist.items():
#         bucket_size = int(ratio * total_samples)
#         candidates = train_buckets.get(b, [])
#         if len(candidates) > bucket_size:
#             new_train.extend(random.sample(candidates, bucket_size))
#         else:
#             new_train.extend(candidates)  # 不够则欠采样（也可选择过采样）
#     return new_train



def my_load_dataset(path):
    dataset = []
    # path = os.path.join(data_pair, f"{lang}.devtest")
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(line.strip())
    return dataset

def merge_entities(tokens):
    merged = []
    current_entity = None

    for tok in tokens:
        ent_type = tok["entity"].split("-")[-1]  # PERS, ORG, LOC
        prefix = tok["entity"].split("-")[0]

        # 如果是B开头，或者和当前实体类型不同，则开启新的实体
        if prefix == "B" or current_entity is None or current_entity["entity"] != ent_type:
            # 先把旧的保存
            if current_entity:
                merged.append(current_entity)

            # 新建一个实体
            current_entity = {
                "entity": ent_type,
                "score": tok["score"],
                "start": tok["start"],
                "end": tok["end"],
                "word": tok["word"].replace("##", "")
            }
        else:
            # I开头，继续拼接
            current_entity["word"] += tok["word"].replace("##", "")
            current_entity["end"] = tok["end"]
            current_entity["score"] = (current_entity["score"] + tok["score"]) / 2  # 平均分可选

    # 别忘了加最后一个
    if current_entity:
        merged.append(current_entity)

    return merged

if __name__ == "__main__":
    lang_code = "sw"
    spacy_dict = {
        "en": "en_core_web_sm",
    }
    bound_dict = {
        "ar": (20, 80),
        "bn": (50, 250),
        "bs": (10, 110),
        "bg": (20, 140),
        "cs": (20, 120),
        "de": (20, 90),
        "tr": (20, 80),
        "hi": (50, 230),
        "en": (10, 50),
        "es": (10, 100),
        "et": (10, 110),
        "fi": (20, 100),
        "mk": (30, 120),
        "id": (10, 100),
        "hu": (20, 120),
        "ru": (30, 180),
        "is": (20, 110),
        "fr": (10, 120),
        "it": (20, 100),
        "nl": (20, 100),
        "pl": (20, 100),
        "pt": (20, 100),
        "ro": (20, 100),
        "ru": (20, 100),
        "sw": (20, 100),
        "uk": (20, 150),
        "zh": (10, 150),
    }
    ner_model_path_dict = {
        "ar": "/mnt/gemini/data1/yifengliu/model/bert-base-arabic-camelbert-msa-ner",
        "hi": "/mnt/gemini/data1/yifengliu/model/IndicNER",
        "tr": "/mnt/gemini/data1/yifengliu/model/bert-base-turkish-ner-cased",
    }
    lower_bound, upper_bound = bound_dict[lang_code]
    path = f"/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/{lang_code}/news.2024.{lang_code}.shuffled.deduped"
    save_path = f"/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/{lang_code}/ner-{lang_code}1m.jsonl"
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
        if lower_bound <= len(tokens['input_ids']) <= upper_bound:
            new_dataset.append({"src": line.strip(), "data_source": f"wmt24_news_crawl_{lang_code}", "length": len(tokens['input_ids'])})
        if len(new_dataset) >= 10_0000:
        # if len(new_dataset) >= 10000:
            break
    
    # if lang_code != "en":
    #     ner_model_path = ner_model_path_dict[lang_code]
    #     ner_model = pipeline("ner", model=ner_model_path, device=0)
    # else:
    #     ner_model = spacy.load(spacy_dict[lang_code])
    # newer_dataset = []
    # # ratio_list = []
    # for i in tqdm(range(len(new_dataset))):
    #     data = new_dataset[i]
    #     original_length = data['length']
    #     data_info = ner_model(data['src'])
    #     # import code; code.interact(local=locals())
    #     if lang_code != "en":
    #         merged_word_info = merge_entities(data_info)
    #         word_list = [word_info['word'] for word_info in merged_word_info]
    #     else:
    #         try:
    #             word_list = [ent.text for ent in data_info.ents if ent.label_ in ["PERSON", "ORG", "FAC", "GPE", "LOC", "WORK_OF_ART"]]
    #         except:
    #             print(f"word_list corrupted")
    #             import code; code.interact(local=locals())
    #     words = " ".join(word_list)
    #     word_tokenized = tokenizer(words)
    #     length = len(word_tokenized['input_ids'])
    #     # ratio_list.append(length/original_length)
    #     if length / original_length < 0.6:
    #         newer_dataset.append(data)
    #     if len(newer_dataset) == 100_0000:
    #         break
    import code; code.interact(local=locals())
    with open(save_path, 'w') as f:
        for data in new_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
