import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from collections import Counter, defaultdict
import random
import spacy

def get_length_distribution(dataset, binsize=100):
    """统计数据集的长度分布（按桶）"""
    # lengths = [len(s.split()) for s, _ in dataset]
    token_lengths = [len(tokenizer(s)['input_ids']) for s in dataset]
    buckets = [l // binsize for l in token_lengths]  # 例如长度 1-10 -> 0 桶
    counter = Counter(buckets)
    total = sum(counter.values())
    dist = {k: v / total for k, v in counter.items()}
    return dist, buckets

def sample_trainset(trainset, test_dist, binsize=100):
    """根据测试集的长度分布，从训练集中采样"""
    # 训练集按桶分类
    train_buckets = defaultdict(list)
    for data in trainset:
        # b = (len(s.split()) - 1) // binsize
        token_length = len(tokenizer(data)['input_ids'])
        train_buckets[token_length//binsize].append(data)

    # 总训练样本数（按比例决定）
    total_samples = sum(len(v) for v in train_buckets.values())
    
    new_train = []
    for b, ratio in test_dist.items():
        bucket_size = int(ratio * total_samples)
        candidates = train_buckets.get(b, [])
        if len(candidates) > bucket_size:
            new_train.extend(random.sample(candidates, bucket_size))
        else:
            new_train.extend(candidates)  # 不够则欠采样（也可选择过采样）
    return new_train

if __name__ == "__main__":
    tgt_lang = "swh"
    dic = {
        "ben": "bn",
        "bul": "bg",
        "bos": "bs",
        "ces": "cs",
        "deu": "de",
        "spa": "es",
        "est": "et",
        "fin": "fi",
        "fra": "fr",
        "ara": "ar",
        "hin": "hi",
        "eng": "en",
        "hun": "hu",
        "ind": "id",
        "isl": "is",
        "ita": "it",
        "nld": "nl",
        "mkd": "mk",
        "pol": "pl",
        "por": "pt",
        "ron": "ro",
        "rus": "ru",
        "swh": "sw",
        "tur": "tr",
        "ukr": "uk",
        "zho_simpl": "zh",
    }
    path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{tgt_lang}.devtest"
    # path = f"/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/{dic[tgt_lang]}/news.2024.{dic[tgt_lang]}.shuffled.deduped"
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[:100000]
    length_distribution = []
    for line in lines:
        input_ids = tokenizer(line)['input_ids']
        length = len(input_ids)
        length_distribution.append(length)
    # import code; code.interact(local=locals())
    plt.hist(length_distribution, bins=[i for i in range(0, 300, 10)], edgecolor='black')

    plt.xlabel('Token Length')
    plt.ylabel('Count')
    plt.title('Token Length Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.show()
    if 'flores' in path:
        plt.savefig(f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores-{tgt_lang}-token.png")
    else:
        plt.savefig(f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores-{tgt_lang}-token2.png")
    # print(f"Token Distribution")