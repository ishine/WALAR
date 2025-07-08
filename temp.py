import json
import random
import pandas as pd
import re

def load_dastset(file_path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

if __name__ == "__main__":
    # Example usage
    lang_dict = {
        "ary-fra": "Arabic-French",
        "eng-arz": "English-Arabic",
        "eng-fra": "English-French",  
        "eng-hau": "English-Hausa",
        "eng-ibo": "English-Igbo",
        "eng-kik": "English-Kikuyu",
        "eng-luo": "English-Luo",
        "eng-som": "English-Somali",
        "eng-swh": "English-Swahili",
        "eng-twi": "English-Twi",
        "eng-xho": "English-Xhosa",
        "eng-yor": "English-Yoruba",
        "yor-eng": "Yoruba-English",
    }
    pair_list = ["ary-fra",
      "eng-arz",
      "eng-fra",
      "eng-hau",
      "eng-ibo",
      "eng-kik",
      "eng-luo",
      "eng-som",
      "eng-swh",
      "eng-twi",
      "eng-xho",
      "eng-yor",
      "yor-eng",]
    for pair in pair_list:
        lang_pair = lang_dict[pair]
        path = f"/mnt/gemini/data1/yifengliu/data/afriMTE/AfriMTE-ade-devtest-v2.{pair}.jsonl"
        new_path = f"/mnt/gemini/data1/yifengliu/data/afriMTE/AfriMTE-ade-devtest-v2.{pair}2.jsonl"
        dataset = load_dastset(path)
        new_dataset = []
        for data in dataset:
            data['src_lang'] = lang_pair.split("-")[0]
            data['tgt_lang'] = lang_pair.split("-")[1]
            new_dataset.append(data)
        with open(new_path, 'w') as f:
            for data in new_dataset:
                f.write(json.dumps(data) + "\n")
    # path = "/mnt/gemini/data1/yifengliu/data/afriMTE/AfriMTE-ade-devtest-v2.ary-fra.jsonl"
    # # save_path = "/mnt/gemini/data1/yifengliu/data/IndicMT/collated/maithili2.jsonl"
    # dataset = load_dastset(path)
    
    # src_pattern = r"<\|im_start\|>assistant\s*<think>(.*?)</think>\s*(.*?)<\|im_end\|>"
    # queries = ["<|im_start|>user\nSunSport also revealed this week that Sunderland legend Jermain Defoe has thrown his hat into the ring for the top job.\nTranslate from English to Tamil:<|im_end|>\n<|im_start|>assistant"]
    # queries = ["<|im_start|>user\nI think if someone had acted like that towards me when I was a sophomore in high school, I would have been really weirded out.\nTranslate from English to Japanese:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n私は高校2年生だったとき、誰かがそんなように行動していたら、すごく戸端ずれになったかなと思います。<|im_end|>"]
    # srcs = [re.search(src_pattern, q, re.DOTALL).group(2).strip() for q in queries]
    # for data in dataset:
    #     data['src_lang'] = "English"
    #     data['tgt_lang'] = "Maithili"
    # with open(save_path, 'w') as f:
    #     for data in dataset:
    #         f.write(json.dumps(data) + "\n")
    
    # import fasttext
    # lang_detect_model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
    
    # lang_info = lang_detect_model.predict(tgts)
    # detect_rewards = [True if language[0].replace("__label__", "") == "zh" else False for language in lang_info[0]]
    # print("Percentage: {:.2f}%".format(detect_rewards.count(True) / len(detect_rewards) * 100))
    
    
    
    # Load the language ID model
    # file_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen2.5-0.5B-En-Zh-1M-bsz128/global_step140_hf/eng-fra.txt"
    # dataset = []
    # with open(file_path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines[:-2]:
    #         dataset.append(json.loads(line))
    
    # tgts = [data['pred'] for data in dataset]
    
    
    
    # prediction = model.predict(sentence)
    

    import code; code.interact(local=locals())