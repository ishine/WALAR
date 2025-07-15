import random
import json
''
support_list = ["af", "als", "am", "an", "ar", "arz", "as", "ast", "av", "az", "azb", "ba", "bar", "bcl", "be", "bg", "bh", "bn", "bo", "bpy", "br", "bs", "bxr", "ca", "cbk", "ce", "ceb", "ckb", "co", "cs", "cv", "cy", "da", "de", "diq", "dsb", "dty", "dv", "el", "eml", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "frr", "fy", "ga", "gd", "gl", "gn", "gom", "gu", "gv", "he", "hi", "hif", "hr", "hsb", "ht", "hu", "hy", "ia", "id", "ie", "ilo", "io", "is", "it", "ja", "jbo", "jv", "ka", "kk", "km", "kn", "ko", "krc", "ku", "kv", "kw", "ky", "la", "lb", "lez", "li", "lmo", "lo", "lrc", "lt", "lv", "mai", "mg", "mhr", "min", "mk", "ml", "mn", "mr", "mrj", "ms", "mt", "mwl", "my", "myv", "mzn", "nah", "nap", "nds", "ne", "new", "nl", "nn", "no", "oc", "or", "os", "pa", "pam", "pfl", "pl", "pms", "pnb", "ps", "pt", "qu", "rm", "ro", "ru", "rue", "sa", "sah", "sc", "scn", "sco", "sd", "sh", "si", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "tyv", "ug", "uk", "ur", "uz", "vec", "vep", "vi", "vls", "vo", "wa", "war", "wuu", "xal", "xmf", "yi", "yo", "yue", "zh"]

def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

if __name__ == "__main__":
    lang_list = ["ltz", "ast", "oci", "bos", "hrv", "mkd", "pol", "srp", "slk", "slv", "ben", "guj", "hin", "mar", "ory", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav"]
    lang_list = ["ltz", "ast", "oci", "bos", "hrv", "mkd", ]
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