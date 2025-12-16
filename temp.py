import json
import random
import pandas as pd
import re
import csv
import os
import sacrebleu
import fasttext
import shutil
import matplotlib.pyplot as plt
import sys
import hanlp
import tqdm
from tqdm import *
sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
from utils import lang_dict, mm_dict, lang_dict, lang2long, long2lang, my_load_dataset, training_langs2, flores_langs, two2three
import masklid
from masklid import MaskLID
# import masklid2
# from masklid2 import MaskLID
from bleurt import score
from sacrebleu.metrics import BLEU, CHRF, TER
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from comet import load_from_checkpoint, download_model

def load_dataset(file_path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                dataset.append(json.loads(line.strip()))
            except:
                break
    return dataset

def load_flores(file_path):
    """
    Load a FLORES file, where each line is a sentence.
    Returns a list of sentences (stripped of whitespace).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def get_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    hyps = [hyp.strip() for hyp in hyps]
    refs = [ref.strip() for ref in refs]
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores101", force=True).score
    return result


def get_spBLEU2(hyps, refs):
    if len(hyps) != len(refs):
        return None
    hyps = [hyp.strip() for hyp in hyps]
    refs = [ref.strip() for ref in refs]
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200", force=True).score
    return result

def get_sentence_bleu(hyp, ref):
    hyp = hyp.strip()
    ref = ref.strip()
    result = sacrebleu.sentence_bleu(hyp, [ref], smooth_method='floor', tokenize="flores101").score
    return result

def calculate_comet_score(src_texts, references, predictions, model_path="/mnt/gemini/data1/yifengliu/model/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"):
    """Calculate COMET score."""
    model = load_from_checkpoint(model_path)
    
    # Prepare inputs for COMET
    inputs = [{"src": src.strip(), "mt": mt.strip(), "ref": ref.strip()} for src, mt, ref in zip(src_texts, predictions, references)]
    
    output = model.predict(inputs, batch_size=16, gpus=1)
    
    scores, mean_score = output.scores, output.system_score
    return {"mean_score": mean_score, "scores": scores}

def get_ratio_list(model, flores_dataset, full_language, tokenizer):
    # flores_glotlid = ['__label__eng_Latn', '__label__deu_Latn', '__label__isl_Latn', '__label__ltz_Latn', '__label__bel_Cyrl', '__label__ces_Latn', '__label__mkd_Cyrl', '__label__pol_Latn', '__label__srp_Cyrl', '__label__slk_Latn', '__label__slv_Latn', '__label__ukr_Cyrl', '__label__ben_Beng', '__label__guj_Gujr', '__label__hin_Deva', '__label__mar_Deva', '__label__npi_Deva', '__label__pan_Guru', '__label__urd_Arab', '__label__hye_Armn', '__label__ell_Grek', '__label__lvs_Latn', '__label__lit_Latn', '__label__fas_Arab', '__label__cym_Latn', '__label__ceb_Latn', '__label__jav_Latn', '__label__arb_Arab', '__label__azj_Latn', '__label__kaz_Cyrl', '__label__tur_Latn', '__label__uzn_Latn', '__label__kan_Knda', '__label__mal_Mlym', '__label__tam_Taml', '__label__tel_Telu', '__label__mya_Mymr', '__label__ekk_Latn', '__label__fin_Latn', '__label__hun_Latn', '__label__kat_Geor', '__label__heb_Hebr', '__label__khm_Khmr', '__label__kor_Hang', '__label__lao_Laoo', '__label__fil_Latn']
    answers = []
    for i in tqdm(range(len(flores_dataset))):
        text = flores_dataset[i]
        ans = model.predict_codeswitch(text, beta = 20 , alpha = 3, max_lambda = 3, min_length = 10, min_prob = 0.90, max_retry=10, alpha_step_increase = 3, beta_step_increase = 5)
        ans = {key.replace("__label__", ""): value for key, value in ans.items()}
        long_lang_id = lang2long.get(full_language, None)
        if long_lang_id is None:
            raise ValueError(f"Language code {lang} not found in lang2long.")
        lang_translation = ans.get(long_lang_id, None)
        if lang_translation is None:
            lang_translation = ""
        answers.append(lang_translation)
    original_token_length = [len(tokenizer(text)['input_ids']) for text in flores_dataset]
    detect_token_length = [len(tokenizer(answer)['input_ids']) for answer in answers]
    ratio_list = [detect_len / orig_len for detect_len, orig_len in zip(detect_token_length, original_token_length) if orig_len > 0]
    # import code; code.interact(local=locals())
    return ratio_list
    

if __name__ == "__main__":
    # xcomet_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    # src_lang = "ara"
    # tgt_lang_list = ['afr', 'amh', 'ara', 'hye', 'asm', 'ast', 'azj', 'bel', 'ben', 'bos', 'bul', 'mya', 'cat', 'ceb', 'zho_simpl', 'hrv', 'ces', 'dan', 
    # 'nld', 'eng', 'est', 'tgl', 'fin', 'fra', 'ful', 'glg', 'lug', 'kat', 'deu', 'ell', 'guj', 'hau', 'heb', 'hin', 'hun', 'isl', 'ibo', 
    # 'ind', 'gle', 'ita', 'jpn', 'jav', 'kea', 'kam', 'kan', 'kaz', 'khm', 'kor', 'kir', 'lao', 'lav', 'lin', 'lit', 'luo', 'ltz', 'mkd', 
    # 'msa', 'mal', 'mlt', 'mri', 'mar', 'mon', 'nob', 'npi', 'nso', 'nya', 'oci', 'ory', 'orm', 'pus', 'fas', 'pol', 'por', 'pan', 'ron', 'rus', 
    # 'srp', 'sna', 'snd', 'slk', 'slv', 'som', 'ckb', 'spa', 'swh', 'swe', 'tgk', 'tam', 'tel', 'tha', 'tur', 'ukr', 'umb', 'urd', 'uzb', 
    # 'vie', 'cym', 'wol', 'xho', 'yor', 'zul']
    # bleu_list = []
    # for tgt_lang in tgt_lang_list:
    #     path = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/nllb-200-distilled-1.3B/{src_lang}-{tgt_lang}.txt"
    #     dataset = load_dataset(path)
    #     hyps = [data['pred'] for data in dataset]
    #     refs = [data['ref'] for data in dataset]
    #     bleu = get_spBLEU(hyps, refs)
    #     bleu_list.append(bleu)
    #     print(f"{tgt_lang}: {bleu}")
    # print(f"Avg Bleu: {sum(bleu_list)/len(bleu_list)}")

    src_lang, tgt_lang = "zh", "kn"
    split_number = 1000

    src_dataset = load_flores(f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{two2three[src_lang]}.devtest")
    tgt_dataset = load_flores(f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{two2three[tgt_lang]}.devtest")
    en_dataset = load_flores(f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/eng.devtest")

    original_model_output_path = f"/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/LLaMAX3-8B-Alpaca/flores/result_{src_lang}-{tgt_lang}.json"
    tuned_model_output_path = f"/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule1024-LlamaX3-8B-schedule_mix10k-1M-bsz128_global_step1800_hf/flores/result_{src_lang}-{tgt_lang}.json"

    with open(original_model_output_path, 'r') as f:
        original_output = json.load(f)['outputs']
    # original_output = json.load(original_model_output_path)['outputs']
    with open(tuned_model_output_path, 'r') as f:
        tuned_output = json.load(f)['outputs']
    # dataset = [{"src": src, "original_output": output1, "tuned_output": output2, "ref": ref} for src, output1, output2, ref in zip(src_dataset, original_output, tuned_output, tgt_dataset)]
    dataset = []
    for src, o, en, t, ref in zip(src_dataset, original_output, en_dataset, tuned_output, tgt_dataset):
        pair = [(o, "original"), (t, "tuned")]
        random.shuffle(pair)
        (out1, tag1), (out2, tag2) = pair
        if src_lang == "en":
            dataset.append({
                "src": src,
                "output1": out1,
                "output2": out2,
                "ref": ref,
                "order": f"{tag1}-first"
            })
        else:
            dataset.append({
                "src": src,
                "en_ref": en,
                "output1": out1,
                "output2": out2,
                "ref": ref,
                "order": f"{tag1}-first"
            })
    random.shuffle(dataset)
    dataset = dataset[:split_number]
    for i in range(1, 11):
        path = f"/mnt/gemini/data1/yifengliu/qe-lr/annotation/annotation_{src_lang}-{tgt_lang}_{i}/dataset.tsv"
        dir_path = os.path.dirname(path)
        os.makedirs(os.path.dirname(path), exist_ok=True) 
            
        with open(path, "w", encoding="utf-8", newline="") as f:
            subdataset = dataset[(i-1)*50: i*50]
            if src_lang == "en":
                writer = csv.DictWriter(f, fieldnames=["src", "output1", "output2", "ref", "order"], delimiter="\t")
            else:
                writer = csv.DictWriter(f, fieldnames=["src", "en_ref", "output1", "output2", "ref", "order"], delimiter="\t")
            writer.writeheader()
            # import code; code.interact(local=locals())
            writer.writerows(subdataset)
        html_path = os.path.join(dir_path, "annotator_pairwise_tool.html")
        readme_path = os.path.join(dir_path, "README.md")
        shutil.copy("/mnt/gemini/data1/yifengliu/qe-lr/annotation/annotator_pairwise_tool.html", html_path)
        shutil.copy("/mnt/gemini/data1/yifengliu/qe-lr/annotation/README.md", readme_path)
    # dataset = load_dataset("/mnt/gemini/data1/yifengliu/qe-lr/pairwise_annotations.jsonl")
    # tsv_dataset = []
    # with open("/mnt/gemini/data1/yifengliu/qe-lr/annotation/dataset.tsv", 'r') as f:
    #     lines = f.readlines()
    #     for line in lines[1:]:
    #         line_list = line.split('\t')
    #         tsv_dataset.append(line_list[-1])
    # # import code; code.interact(local=locals())
    # original, tuned, tie = 0, 0, 0
    # for data, first in zip(dataset, tsv_dataset):
    #     if data['preference'] == "translation1":
    #         if "tuned" in first:
    #             tuned += 1
    #         elif "original" in first:
    #             original += 1
    #     elif data['preference'] == "translation2":
    #         if "tuned" in first:
    #             original += 1
    #         elif "original" in first:
    #             tuned += 1
    #     else:
    #         tie += 1
    # print(f"original: {original/30}, tuned: {tuned/30}, tie: {tie/30}")
    
    # dataset = []
    # with open(save_path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines[-2]:
    #         dataset.append(line.strip())

    # lang_list = ["en", "ar", "tr", "hi"]
    # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/final_llamax_mix-100k.jsonl"
    # final_mix_dataset = []
    # for lang in lang_list:
    #     path = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/llamax_{lang}-mix-1m.jsonl"
    #     dataset = load_dataset(path)
    #     final_mix_dataset.extend(dataset)
    # random.shuffle(final_mix_dataset)
    # # # # import code; code.interact(local=locals())
    # with open(save_path, 'w') as f:
    #     for data in final_mix_dataset:
    #         f.write(json.dumps(data) + "\n")
        
    # srcs = [data['src'] for data in dataset]
    # preds = [data['pred'] for data in dataset]
    # refs = [data['ref'] for data in dataset]
        
    # bleu = BLEU(tokenize='flores200', smooth_method='exp', lowercase=False)
    # bleu_score = bleu.corpus_score(preds, [refs])
    # print(bleu_score)
    import code; code.interact(local=locals())