import os
import random
import json
import sys
import tqdm
from tqdm import *
from datasets import Dataset, concatenate_datasets
sys.path.append('/mnt/gemini/data1/yifengliu/qe-lr/code')
from utils import three2two, training_langs2, mm_dict, lang_dict, flores_langs, qwen_langs
support_list = ["af", "als", "am", "an", "ar", "arz", "as", "ast", "av", "az", "azb", "ba", "bar", "bcl", "be", "bg", "bh", "bn", "bo", "bpy", "br", "bs", "bxr", "ca", "cbk", "ce", "ceb", "ckb", "co", "cs", "cv", "cy", "da", "de", "diq", "dsb", "dty", "dv", "el", "eml", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "frr", "fy", "ga", "gd", "gl", "gn", "gom", "gu", "gv", "he", "hi", "hif", "hr", "hsb", "ht", "hu", "hy", "ia", "id", "ie", "ilo", "io", "is", "it", "ja", "jbo", "jv", "ka", "kk", "km", "kn", "ko", "krc", "ku", "kv", "kw", "ky", "la", "lb", "lez", "li", "lmo", "lo", "lrc", "lt", "lv", "mai", "mg", "mhr", "min", "mk", "ml", "mn", "mr", "mrj", "ms", "mt", "mwl", "my", "myv", "mzn", "nah", "nap", "nds", "ne", "new", "nl", "nn", "no", "oc", "or", "os", "pa", "pam", "pfl", "pl", "pms", "pnb", "ps", "pt", "qu", "rm", "ro", "ru", "rue", "sa", "sah", "sc", "scn", "sco", "sd", "sh", "si", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "tyv", "ug", "uk", "ur", "uz", "vec", "vep", "vi", "vls", "vo", "wa", "war", "wuu", "xal", "xmf", "yi", "yo", "yue", "zh"]

tgt_lang_dict = {
    "ara": ['amh', 'asm', 'bel', 'est', 'fin', 'guj', 'hau', 'hun', 'hye', 'ibo', 'isl', 'jav', 'kat',  'kaz', 'kan', 'kor', 'kir', 'ltz', 'lit', 'mri', 'mal', 'mon', 'mar', 'nob', 'ory', 'pan', 'pol', 'pus', 'snd', 'slv', 'sna', 'som', 'srp', 'swh', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    "ben": ['afr', 'amh', 'ara', 'hye', 'azj', 'bel', 'bul', 'ceb', 'zho_simpl', 'ces', 'dan', 'nld', 'est', 'tgl', 'fin', 'glg', 'kat', 'deu', 'ell', 'hau', 'heb', 'hun', 'isl', 'ibo', 'gle', 'ita', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'msa', 'mal', 'mri', 'mar', 'mon', 'nob', 'npi', 'pus', 'fas', 'pol', 'rus', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'spa', 'swh', 'swe', 'tgk', 'tam', 'tur', 'ukr', 'urd', 'uzb', 'cym', 'xho', 'yor', 'zul'],
    "bul": ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'est', 'fin', 'guj', 'hau', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'pan', 'srp', 'sna', 'snd', 'som', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    "ces": ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'est', 'fin', 'guj', 'hau', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'pan', 'srp', 'sna', 'snd', 'som', 'swh', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    "deu": ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'est', 'guj', 'hau', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'pan', 'srp', 'sna', 'snd', 'som', 'swh', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    "eng": ['amh', 'asm', 'bel', 'hau', 'ibo', 'jav', 'mri', 'mon', 'mya', 'nso', 'ory', 'pus', 'snd', 'sna', 'som', 'srp', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    "fin": ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'est', 'tgl', 'kat', 'ell', 'guj', 'hau', 'heb', 'hin', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'msa', 'mal', 'mri', 'mar', 'mon', 'nob', 'npi', 'pus', 'fas', 'pol', 'pan', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'swh', 'tgk', 'tam', 'tel', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    "fra": ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'est', 'fin', 'guj', 'hau', 'isl', 'ibo', 'jav', 'kan', 'kaz', 'kir', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'pan', 'srp', 'sna', 'snd', 'som', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    "hin": ['amh', 'ara', 'azj', 'bel', 'ceb', 'zho_simpl', 'ces', 'nld', 'est', 'tgl', 'fin', 'kat', 'ell', 'hau', 'heb', 'hun', 'isl', 'ibo', 'gle', 'ita', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'mal', 'mri', 'mar', 'mon', 'nob', 'pus', 'fas', 'pol', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'spa', 'swh', 'tgk', 'tam', 'tur', 'ukr', 'uzb', 'xho', 'yor', 'zul'],
    "hun": ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'est', 'tgl', 'fin', 'kat', 'ell', 'guj', 'hau', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'mal', 'mri', 'mar', 'mon', 'nob', 'npi', 'pus', 'fas', 'pol', 'pan', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'swh', 'tgk', 'tam', 'tel', 'tur', 'urd', 'uzb', 'cym', 'xho', 'yor', 'zul'],
    "ind": ['amh', 'ara', 'azj', 'bel', 'ceb', 'zho_simpl', 'est', 'fin', 'kat', 'guj', 'hau', 'hun', 'isl', 'ibo', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'pol', 'pan', 'srp', 'sna', 'snd', 'slv', 'som', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    "ita": ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'est', 'fin', 'kat', 'guj', 'hau', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'fas', 'pol', 'pan', 'srp', 'sna', 'snd', 'slv', 'som', 'swh', 'tam', 'tel', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    "isl": ['amh', 'ara', 'hye', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'ces', 'est', 'tgl', 'fin', 'kat', 'ell', 'guj', 'hau', 'heb', 'hin', 'hun', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'msa', 'mal', 'mri', 'mar', 'mon', 'nob', 'npi', 'pus', 'fas', 'pol', 'pan', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'spa', 'swh', 'tgk', 'tam', 'tel', 'tur', 'ukr', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    'mkd': ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'est', 'fin', 'guj', 'hau', 'hun', 'isl', 'ibo', 'jav', 'kan', 'kaz', 'kir', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'pan', 'srp', 'sna', 'snd', 'som', 'swh', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    'nld': ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'est', 'tgl', 'fin', 'kat', 'ell', 'guj', 'hau', 'heb', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'fas', 'pol', 'pan', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'swh', 'tgk', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    'pol': ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'est', 'tgl', 'fin', 'kat', 'ell', 'guj', 'hau', 'heb', 'hin', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'mal', 'mri', 'mar', 'mon', 'nob', 'npi', 'pus', 'fas', 'pan', 'srp', 'sna', 'snd', 'slv', 'som', 'swh', 'tgk', 'tam', 'tel', 'tur', 'urd', 'uzb', 'cym', 'xho', 'yor', 'zul'],
    'por': ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'est', 'fin', 'hau', 'isl', 'ibo', 'jav', 'kan', 'kaz', 'kir', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'pan', 'srp', 'sna', 'snd', 'som', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    'ron': ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'est', 'fin', 'guj', 'hau', 'isl', 'ibo', 'jav', 'kan', 'kaz', 'kir', 'lit', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'pan', 'srp', 'sna', 'snd', 'som', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    'rus': ['amh', 'ara', 'azj', 'ben', 'ceb', 'zho_simpl', 'est', 'fin', 'guj', 'hau', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'ltz', 'mal', 'mri', 'mar', 'mon', 'nob', 'npi', 'pus', 'pan', 'srp', 'sna', 'snd', 'som', 'swh', 'tam', 'tel', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    'spa': ['amh', 'ara', 'asm', 'azj', 'bel', 'ben', 'bos', 'ceb', 'zho_simpl', 'hrv', 'est', 'tgl', 'fin', 'kat', 'guj', 'hau', 'heb', 'hin', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kea', 'kam', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'mal', 'mri', 'mar', 'mon', 'nob', 'npi', 'nya', 'oci', 'ory', 'pus', 'fas', 'pol', 'pan', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'swh', 'tgk', 'tam', 'tel', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    'tur': ['amh', 'ara', 'azj', 'bel', 'ben', 'ceb', 'zho_simpl', 'ces', 'est', 'tgl', 'fin', 'kat', 'ell', 'guj', 'hau', 'heb', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'mal', 'mri', 'mar', 'mon', 'nob', 'npi', 'pus', 'fas', 'pol', 'pan', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'swh', 'tgk', 'tam', 'urd', 'uzb', 'cym', 'xho', 'yor', 'zul'],
    'ukr': ['amh', 'ara', 'azj', 'ben', 'ceb', 'zho_simpl', 'est', 'fin', 'guj', 'hau', 'hun', 'isl', 'ibo', 'gle', 'jav', 'kan', 'kaz', 'kir', 'ltz', 'mal', 'mri', 'mar', 'mon', 'npi', 'pus', 'pan', 'srp', 'sna', 'snd', 'som', 'swh', 'tam', 'tur', 'urd', 'uzb', 'xho', 'yor', 'zul'],
    'zho_simpl': ['afr', 'amh', 'ara', 'hye', 'azj', 'bel', 'ben', 'bul', 'ceb', 'ces', 'nld', 'est', 'tgl', 'fin', 'kat', 'deu', 'ell', 'guj', 'hau', 'heb', 'hin', 'hun', 'isl', 'ibo', 'gle', 'ita', 'jav', 'kan', 'kaz', 'kir', 'lav', 'lit', 'ltz', 'mkd', 'msa', 'mal', 'mri', 'mar', 'mon', 'nob', 'npi', 'pus', 'fas', 'pol', 'pan', 'rus', 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'spa', 'swh', 'swe', 'tgk', 'tam', 'tel', 'tur', 'ukr', 'urd', 'uzb', 'cym', 'xho', 'yor', 'zul'],
    
    
}

def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def get_lang(lang_code):
    src_lang = mm_dict.get(lang_code, '')
    if len(src_lang) == 0:
        src_lang = lang_dict.get(lang_code, '')
    if src_lang == '':
        src_lang = lang_code.capitalize()
    return src_lang

def make_prompt(source, src, tgt, model_name, template_type='chat', tokenizer=None):
    if template_type == 'base':
        if model_name == "llamax":
            return f"""Translate the following sentences from {src_lang} to {tgt_lang}.\n### Input:\n{source}\n"""
        else:
            return f"{source}\nTranslate from {src} to {tgt}:"
    elif template_type == 'chat':
        return f"You are a helpful assistant. Translate this text from {src} to {tgt}:\n{source}"
    elif template_type == 'rl':
        return f"Translate this text from {src} to {tgt}:\n{source}"
    else:
        raise ValueError(f"Unknown template type: {template_type}")

if __name__ == "__main__":
    # src_lang_list = ["en"]
    # lang_list = ["isl", "ltz", "bel", "ces", "mkd", "pol", "slk", "slv", "ukr", "guj", "mar", "npi", "pan", "urd", "hye", "ell", "lav", "lit", "fas", "cym", "ceb", "tgl", "jav", "ara", "azj", "tur", "uzb", "kan", "mal", "tam", "est", "fin", "hun", "kat", "heb", "kor", "tha", "eng", "spa", "zho", "ind", "ben", "rus", "tel"]
    # src_lang_list = ["is", "lb", "be", "cs", "mk", "pl", "sk", "sl", "uk", "gu", "mr", "ne", "pa", "ur", "hy", "el", "lv", "lt", "fa", "cy", "ceb", "tl", "jv", "ar", "az", "tr", "uz", "kn", "ml", "ta", "et", "fi", "hu", "ka", "he", "ko", "th", "en", "es", "zh", "id", "bn", "ru", "te"]
    # lang_list = ["eng"]
    # src_lang_list = ["en", "ar", "tr", "hi", "hu", "bg", "id", "es"]
    # src_lang_list = list(tgt_lang_dict.keys())
    src_lang_list = ['ara', 'ben', 'bul', 'ces', 'deu', 'eng', 'fin', 'fra', 'hin', 'hun', 'ind', 'ita', 'isl', 'mkd', 'nld', 'pol', 'por', 'ron', 'rus', 'spa', 'tur', 'ukr', 'zho_simpl']
    src_lang_list2 = ['eng', 'deu', 'fra', 'spa', 'ita', 'por', 'rus', 'nld', 'bul', 'ind', 'ron', 'mkd', 'hin', 'ces', 'zho_simpl', 'fin', 'hun', 'pol', 'tur', 'ukr', 'ben', 'ara', 'isl']
    
    # import code; code.interact(local=locals())
    
    metricx_not_support_list = ["ast", "oci", "bos", "hrv", "asm", "ory", "lug", "kea", "kam", "lin", "nya", "wol", "ful", "orm", "luo"]
    non_segmented_list = ["zho_simpl", "jpn", "zho_trad", "tha", "kor", "khm", "lao", "mya"]
    # lang_list = training_langs2

    # lang_list.append("eng")
    schedule = False
    num_per_lang = 10000
    # src_lang_list = ["en", "ar", "tr", "hi", "hu", "bg", "id", "es"]
    # src_lang_list = ["is", "lb", "be", "cs", "mk", "pl", "sk", "sl", "uk", "gu", "mr", "ne", "pa", "ur", "hy", "el", "lv", "lt", "fa", "cy", "ceb", "tl", "jv", "ar", "az", "tr", "uz", "kn", "ml", "ta", "et", "fi", "hu", "ka", "he", "ko", "th", "en", "es", "zh", "id", "bn", "ru", "te"]
    # uz, bn, ru, te
    # import code; code.interact(local=locals())
    # model_name = "llamax"
    model_name = "llamax"
    output_file = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/schedule_mix10k.jsonl"
    all_datasets = []
    for src_lang in src_lang_list:
        tgt_lang_list = tgt_lang_dict[src_lang]
        # for lang in list(tgt_lang_dict.keys()):
        #     if lang not in tgt_lang_list:
        #         tgt_lang_list.append(lang)
        src_lang = three2two.get(src_lang, src_lang)
        other_lang_list = [lang for lang in flores_langs if lang in metricx_not_support_list and lang not in tgt_lang_list]
        if 'qwen' in model_name.lower():
            other_lang_list = [lang for lang in other_lang_list if lang in qwen_langs]
        meta_file_path = f"/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/{src_lang}/ner-{src_lang}1m.jsonl"
        # meta_file_path = f"/mnt/gemini/data1/yifengliu/data/glot500/{src_lang}/{src_lang}.jsonl"
        print(f"Loading meta dataset from: {meta_file_path}")
        meta_dataset = load_dataset(meta_file_path)
        random.shuffle(meta_dataset)
        # output_file = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/new_eight_directions_{src_lang}-mix-1m.jsonl"
        src_lang = get_lang(src_lang)
        final_lang_list = tgt_lang_list
        # final_lang_list = []
        # for lang in lang_list:
        #     two_lang = three2two[lang]
        #     if two_lang in support_list:
        #         final_lang_list.append(lang)
        # import code; code.interact(local=locals())
        def make_map_fn(split, src_lang, tgt_lang):
            def process_fn(example, idx):
                data_source = example.get('data_source', 'unknown')
                # Dynamic source and target language field extraction
                source = example['src']
                
                prompt = make_prompt(source, src_lang, tgt_lang, model_name, template_type="base")
                
                data = {
                    "data_source": data_source + "_" + f"{src_lang}-{tgt_lang}",
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "lang_pair": f"{src_lang}-{tgt_lang}",
                    "src_text": source,
                    "input_key": prompt,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "ability": "translate",
                    "extra_info": {
                        'index': idx,
                    }
                }
                return data
            return process_fn
        
        # for lang in final_lang_list:
        # all_datasets = []
        for i in tqdm(range(len(final_lang_list))):
            partial_dataset = meta_dataset[num_per_lang * i: num_per_lang * (i + 1)]
            partial_dataset = Dataset.from_list(partial_dataset)
            lang = final_lang_list[i]
            tgt_lang = get_lang(lang)
            if tgt_lang != src_lang:
                train_dataset = partial_dataset.map(
                    function=make_map_fn('train', src_lang, tgt_lang), 
                    with_indices=True
                )
                all_datasets.append(train_dataset)
    # import code; code.interact(local=locals())
    final_dataset = concatenate_datasets(all_datasets)
    
    final_dataset = final_dataset.shuffle(seed=42)
    # import code; code.interact(local=locals())
    if schedule:
        lang_reverse_dict = {v: k for k, v in lang_dict.items()}
        lang_order = {lang: idx for idx, lang in enumerate(src_lang_list2)}
        sorted_dataset = sorted(
            final_dataset,
            key=lambda x: min(lang_order.get(lang_reverse_dict[lang], len(src_lang_list2)) for lang in x['lang_pair'].split('-'))
        )
        final_dataset = sorted_dataset
    dir_name = os.path.dirname(output_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    # 写出 JSON lines 文件（确保中文不转义）
    with open(output_file, "w", encoding="utf-8") as f:
        for example in final_dataset:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")

    # Print dataset format

    # print("Train dataset columns:")
    # train_pdf = final_dataset.to_pandas()
    # print(train_pdf.head())
    # print(train_pdf['prompt'][0])
    
    # print(f"Train dataset saved to: {output_file}")
    import code; code.interact(local=locals())
    