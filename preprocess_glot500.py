import os
import json
import sys
sys.path.append('/mnt/gemini/data1/yifengliu/qe-lr/code')
from utils import lang_dict, long2lang, mm_dict

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

# def my_load_dataset(path):
#     with open(path, 'r') as f:
#         dataset = [json.loads(line) for line in f]
#     return dataset

# training_langs_200 = [
#     # "isl_Latn", "ltz_Latn", "bel_Cyrl", "ces_Latn", "mkd_Cyrl", "pol_Latn", "slk_Latn", "slv_Latn", "ukr_Cyrl",
#     # "guj_Gujr", "hin_Deva", "mar_Deva", "npi_Deva", "pan_Guru", "urd_Arab", "hye_Armn", "ell_Grek", "lit_Latn",
#     # "pes_Arab", "cym_Latn", "ceb_Latn", 
#     # "tgl_Latn", "jav_Latn", "ara_Arab", "azj_Latn", "tur_Latn", "kan_Knda",
#     "mal_Mlym", "tam_Taml", "est_Latn", "fin_Latn", "hun_Latn", "kat_Geor", "heb_Hebr", "kor_Hang", "tha_Thai",
#     "lvs_Latn",
#     "eng_Latn", "spa_Latn", "zho_Hani", "ind_Latn"
# ]
# # news crawl: ben_Beng, tel_Telu,
# # "uzn_Latn"
# model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# mm_dict_reverse = {v: k for k, v in mm_dict.items()}
# # mm_dict_reverse.update({"tl": "Tagalog"})
# mm_dict_reverse['Tagalog']="tl"
# for lang in training_langs_200:
#     dataset = load_dataset('cis-lmu/Glot500', lang, cache_dir="/mnt/gemini/data1/yifengliu/data/glot500", split='train')
#     # import code; code.interact(local=locals())
#     two_character_lang = mm_dict_reverse[long2lang[lang]]
#     save_path = f"/mnt/gemini/data1/yifengliu/data/glot500/{two_character_lang}/{two_character_lang}.jsonl"
#     dirname = os.path.dirname(save_path)
#     if dirname and not os.path.exists(dirname):
#         os.makedirs(dirname)
#     dataset = dataset[:50_0000]
#     cnt = 0
#     with open(save_path, 'w', encoding='utf-8') as f:
#         for text in dataset['text']:
#             for txt in text.split("\n"):
#                 txt = txt.strip()
#                 tokens = tokenizer(txt)
#                 token_length = len(tokens['input_ids'])
#                 if 30 < token_length < 200:
#                     f.write(
#                         json.dumps({
#                             "src": txt,
#                             "data_source": f"glot500_{two_character_lang}",
#                             "length": len(tokens['input_ids'])
#                         }, ensure_ascii=False) + "\n"
#                     )
#                     cnt += 1
#                     break
#             if cnt == 50_0000:
#                 break

# import code; code.interact(local=locals())

import json
path_list = [
    # "/mnt/gemini/data1/yifengliu/data/madlad/bn/clean_docs_v2-00000-of-05000.jsonl",
    "/mnt/gemini/data1/yifengliu/data/madlad/ru/clean_docs_v2-00000-of-05000.jsonl",
    "/mnt/gemini/data1/yifengliu/data/madlad/te/clean_docs_v2-00002-of-05000.jsonl",
    "/mnt/gemini/data1/yifengliu/data/madlad/uz/clean_docs_v2-00001-of-00200.jsonl",
    # "/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/bn/news.2024.bn.shuffled.deduped"
    ]
model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

for path in path_list:
    dirname = os.path.dirname(path)
    # dataset = my_load_dataset(path)
    with open(path, 'r') as f:
        lines = f.readlines()
        dataset = [json.loads(line.strip()) for line in lines]
    lang = path.split("/")[-2]
    save_path = os.path.join(dirname, f"{lang}.jsonl")
    new_dataset = []
    for data in dataset:
        # tokens = tokenizer(data['text'])
        # import code; code.interact(local=locals())
        for dt in data['text'].split("\\n"):
            dt = dt.strip()
            tokens = tokenizer(dt)
            length = len(tokens['input_ids'])
            if 30 < length < 200:
                new_dataset.append({
                    # "src": data['text'],
                    "src": dt,
                    "data_source": f"wmt24_{lang}",
                    "length": length
                })
                # import code; code.interact(local=locals())
                break
        if len(new_dataset) == 50_0000:
            break
    with open(save_path, 'w') as f:
        for data in new_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")