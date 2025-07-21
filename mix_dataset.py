import random
import json
import sys
sys.path.append('/mnt/gemini/data1/yifengliu/qe-lr/code')
from utils import three2two
support_list = ["af", "als", "am", "an", "ar", "arz", "as", "ast", "av", "az", "azb", "ba", "bar", "bcl", "be", "bg", "bh", "bn", "bo", "bpy", "br", "bs", "bxr", "ca", "cbk", "ce", "ceb", "ckb", "co", "cs", "cv", "cy", "da", "de", "diq", "dsb", "dty", "dv", "el", "eml", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "frr", "fy", "ga", "gd", "gl", "gn", "gom", "gu", "gv", "he", "hi", "hif", "hr", "hsb", "ht", "hu", "hy", "ia", "id", "ie", "ilo", "io", "is", "it", "ja", "jbo", "jv", "ka", "kk", "km", "kn", "ko", "krc", "ku", "kv", "kw", "ky", "la", "lb", "lez", "li", "lmo", "lo", "lrc", "lt", "lv", "mai", "mg", "mhr", "min", "mk", "ml", "mn", "mr", "mrj", "ms", "mt", "mwl", "my", "myv", "mzn", "nah", "nap", "nds", "ne", "new", "nl", "nn", "no", "oc", "or", "os", "pa", "pam", "pfl", "pl", "pms", "pnb", "ps", "pt", "qu", "rm", "ro", "ru", "rue", "sa", "sah", "sc", "scn", "sco", "sd", "sh", "si", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "tyv", "ug", "uk", "ur", "uz", "vec", "vep", "vi", "vls", "vo", "wa", "war", "wuu", "xal", "xmf", "yi", "yo", "yue", "zh"]

def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

if __name__ == "__main__":
    # lang_list = ["ltz", "ast", "oci", "bos", "hrv", "mkd", "pol", "srp", "slk", "slv", "ben", "guj", "hin", "mar", "ory", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav"]
    # lang_list = ["ltz", "ast", "oci", "bos", "hrv", "mkd", ]
    lang_list = ["ltz", "mkd", "pol", "srp", "slk", "slv", "ben", "guj", "hin", "mar", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav", "ara", "tur", "tam", "fin"]
    final_lang_list = []
    for lang in lang_list:
        two_lang = three2two[lang]
        if two_lang in support_list:
            final_lang_list.append(lang)
    # import code; code.interact(local=locals())
    
    num_dict = {k: 20000 for k in final_lang_list}
    save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-mix-mid2-1m.jsonl"
    new_dataset = []
    index = 0
    for lang in final_lang_list:
        file_path = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_eng-{lang}-1m.jsonl"
        dataset = load_dataset(file_path)
        new_dataset.extend(dataset[index:index + num_dict[lang]])
        index += num_dict[lang]
    
    random.shuffle(new_dataset)
    with open(save_path, 'w') as f:
        for data in new_dataset:
            f.write(json.dumps(data) + "\n")