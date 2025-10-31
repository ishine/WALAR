import json
import random
import pandas as pd
import re
import os
import sacrebleu
import fasttext
import matplotlib.pyplot as plt
import sys
import hanlp
import tqdm
from tqdm import *
sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
from utils import lang_dict, mm_dict, lang_dict, lang2long, long2lang, my_load_dataset, training_langs2, flores_langs
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
    xcomet_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    # Example usage
    # dataset = []
    # path = "/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/ja/ja1m.jsonl"
    # save_path = "/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/en/en1m.jsonl"
    # dataset = load_dataset(path)
    # new_dataset = []

    # path = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Detect-MetricX-Qwen2.5-0.5B-en-zh-1M-bsz128/global_step780_hf/eng-zho_simpl.txt"
    
    # tgt_list = ["zho_simpl", "ara", "deu", "spa", "fin", "jpn", "rus"]
    # for tgt in tgt_list:
    #     path = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Qwen3-32B-AWQ-DA-Qwen2.5-3B-Instruct-en-zh-1M-bsz128/global_step90_hf/eng-{tgt}.txt"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    #     # dataset = load_dataset(path)
    #     # dataset = load_dataset(path)
    #     # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Detect-MetricX-Qwen2.5-3B-Instruct-en-mix-1M-bsz128/global_step180_hf/eng-zho_simpl.txt"
    #     with open(path, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines[:-2]:
    #             dataset.append(json.loads(line.strip()))
        # srcs = [data['src'] for data in dataset]
        # hyps = [data['pred'] for data in dataset]
        # refs = [data['ref'] for data in dataset]
    #     xcomet_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    #     score = calculate_comet_score(srcs, refs, hyps, model_path=xcomet_path)
    #     print(lines[-2])
    #     print(lines[-1])
    #     print(f"{path}: {score['mean_score']}")
    #     with open(path, 'a') as f:
    #         f.write(f"XCOMET Score: {score['mean_s
    # core']:.4f}\n")
    # Requires transformers>=4.51.0
    # Requires sentence-transformers>=2.7.0


    # 26.0056
    # dataset = []
    # lang_list = ["en", "ar", "tr", "hi"]
    # final_mix_dataset = []
    # # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Final-en-LlamaX3-8B-llamax_en-mix-1m-1M-bsz128/global_step40_hf/eng-srp.txt"
    # # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/LLaMAX3-8B-Alpaca/eng-srp.txt"
    # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/new_llamax_mix-1m.jsonl"
    # for lang in lang_list:
    #     path = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/llamax_{lang}-mix-1m.jsonl"
    #     dataset = load_dataset(path)
    #     final_mix_dataset.extend(dataset)
    # # dataset = load_dataset(save_path)
    # random.shuffle(final_mix_dataset)
    # with open(save_path, 'w') as f:
    #     for data in dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # dataset = []
    # lang_list = ["en", "ar", "tr", "hi", "es", "bg", "id", "hu"]
    # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/new_eight_directions_mix-1m.jsonl"
    # for lang in lang_list:
    #     with open(f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/new_eight_directions_{lang}-mix-1m.jsonl", 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             dataset.append(json.loads(line.strip()))
    # # dataset = dataset[61440:]
    # random.shuffle(dataset)
    # import code; code.interact(local=locals())
    # with open(save_path, 'w') as f:
    #     for data in dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/new_eight_directions_mix-1m.jsonl"
    # dataset = load_dataset(path)
    
    # lang_list = ["hun", "vie", "spa", "ces", "fra", "deu", "rus", "ben", "srp", "kor", "jpn", "tha", "swh", "zho_simpl", "tel", "eng"]
    # bleu_list = []
    # for lang in lang_list:
    #     path = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/LLaMAX3-8B-Alpaca/ara-{lang}.txt"
    #     # path = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/new_eight_directions_{lang}-mix-1m.jsonl"
    #     dataset = load_dataset(path)
    #     preds = [data['pred'] for data in dataset]
    #     preds = [pred.strip().split('\n')[0] for pred in preds]
    #     refs = [data['ref'] for data in dataset]
    #     bleu_list.append(get_spBLEU(preds, refs))
    #     # print(f"{lang}: {len(dataset)}")
    # print("Avg spBLEU:", sum(bleu_list) / len(bleu_list))
    
    # with open("/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/zho_simpl.devtest", 'r') as f:
    #     flores_dataset = f.readlines()
    #     dataset = [data.strip() for data in flores_dataset if data.strip()]
    # tokenizer = hanlp.load("/mnt/taurus/home/yifengliu/.hanlp/tok/coarse_electra_small_20220616_012050", devices=0)
    # dataset = [tokenizer(data) for data in dataset]
    
    # import fasttext
    # model_path = "/mnt/gemini/data1/yifengliu/model/models--cis-lmu--glotlid/snapshots/74cb50b709c9eefe0f790030c6c95c461b4e3b77/model.bin"
    # model = fasttext.load_model(model_path)
    # # lang_list = ["amh", "azj", "bel", "hau", "ibo", "isl", "jav", "kan", "kor", "kir", "lit", "mri", "mal", "mon", "mar", "nso", "pol", "pus", "snd", "sna", "som", "srp", "tam", "tha", "tur", "umb", "urd", "uzb", "xho", "yor", "zul"]
    # lang_list = flores_langs
    # for lang in lang_list:
    #     path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{lang}.devtest"
    #     with open(path, 'r') as f:
    #         flores_dataset = f.readlines()
    #         dataset = [data.strip() for data in flores_dataset if data.strip()]
    #     lang_info = model.predict(dataset[0])
    #     long_lang = lang_info[0][0].replace("__label__", "")
    #     full_lang = long2lang.get(long_lang, "unknown")
    #     lang_reverse_dict = {v: k for k, v in lang_dict.items()}    
    #     full_language = lang_reverse_dict.get(full_lang, "unknown") 
                    
    #     if full_language != lang:
    #         print(f"Language mapping error: {lang} -> {full_language}")
    
    # dataset = []
    # with open("/mnt/gemini/data1/yifengliu/qe-lr/data/train/schedule.jsonl", "r") as f:
    #     lines = f.readlines()
    #     for data in lines:
    #         dataset.append(json.loads(data.strip()))
    # import code; code.interact(local=locals())
    # dataset = dataset[128*550:]
    # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/schedule2.jsonl"
    # with open(save_path, 'w') as f:
    #     for data in dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    
    
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128/global_step300_hf/eng-kan.txt"
    # tokenizer = AutoTokenizer.from_pretrained("/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca")
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/final_tr-mix-1m.jsonl"
    # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/final2_tr-mix-1m.jsonl"
    # dataset = load_dataset(path)
    # dataset = dataset[19200:]
    # with open(save_path, 'w') as f:
    #     for data in dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")
    # src_lang = "eng"
    # tgt_lang_list = training_langs2
    # tgt_lang_list = ["ltz", "bel", "ces", "mkd" ,"pol" ,"srp" ,"slk" ,"slv" ,"ukr", "ben" ,"guj" ,"hin" ,"mar" ,"npi" ,"pan" ,"urd" ,"hye" ,"ell" ,"lav" ,"lit", "fas", "cym" ,"ceb" ,"tgl" ,"jav", "azj", "kaz", "tur", "uzb", "kan", "mal" ,"tam", "tel", "mya", "est", "fin", "hun","kat","heb","khm","kor","lao","tha"]
    # tgt_lang_list = ['afr', 'dan', 'nld', 'deu', 'nob', 'swe', 'cat', 'fra', 'glg', 'por', 'ron', 'spa', 'bul', 'rus', 'ita', 'ind', 'msa', 'zho_simpl', 'jpn', 'vie']
    # # tgt_lang_list = ["ces"]
    # for tgt in tgt_lang_list:
    #     file_path = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Pure2-QE-Qwen3-4B-base_en-mix-mid2-1m-bsz128/global_step900_hf/{src_lang}-{tgt}.txt"
    #     with open(file_path, 'r') as f:
    #         lines = f.readlines()
            
    #     if lines:
    #         lines = lines[:-1]
    #     with open(file_path, 'w') as f:
    #         f.writelines(lines)
    
    # with open("/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen2.5-0.5B-En-Zh-1M-bsz128/global_step360_hf", 'r') as f:
    # dataset = load_dataset("/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen2.5-0.5B-En-Zh-1M-bsz128/global_step360_hf/eng-zho_simpl.txt")    
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/final_mix-160k.jsonl"
    # dataset = load_dataset(path)
    
    flores_glotlid = ['__label__rus_Cyrl', '__label__eng_Latn', '__label__arb_Arab', '__label__rus_Cyrl', '__label__por_Latn', '__label__pol_Latn', '__label__ekk_Latn', '__label__ell_Grek', '__label__slk_Latn', '__label__slv_Latn', '__label__nld_Latn', '__label__lvs_Latn', '__label__hun_Latn', '__label__dan_Latn', '__label__swe_Latn', '__label__lit_Latn', '__label__fin_Latn', '__label__mlt_Latn', '__label__cmn_Hani', '__label__nob_Latn', '__label__kor_Hang', '__label__ind_Latn', '__label__uzn_Latn', '__label__fil_Latn', '__label__ukr_Cyrl', '__label__hin_Deva', '__label__hin_Latn', '__label__afr_Latn', '__label__mar_Deva', '__label__ceb_Latn', '__label__ilo_Latn', '__label__zul_Latn', '__label__heb_Hebr', '__label__xho_Latn', '__label__vie_Latn', '__label__jpn_Jpan', '__label__guj_Gujr', '__label__hrv_Latn', '__label__tur_Latn', '__label__nya_Latn', '__label__tsn_Latn', '__label__sna_Latn', '__label__tso_Latn', '__label__tha_Thai', '__label__spa_Latn', '__label__deu_Latn', '__label__eus_Latn', '__label__bul_Cyrl', '__label__amh_Ethi', '__label__fra_Latn', '__label__ewe_Latn', '__label__mkd_Cyrl', '__label__nso_Latn', '__label__tam_Taml', '__label__lin_Latn', '__label__twi_Latn', '__label__yor_Latn', '__label__als_Latn', '__label__ibo_Latn', '__label__ben_Beng', '__label__ita_Latn', '__label__tpi_Latn', '__label__azj_Latn', '__label__run_Latn', '__label__mya_Mymr', '__label__kin_Latn', '__label__ron_Latn', '__label__ces_Latn', '__label__kat_Geor', '__label__urd_Arab', '__label__zsm_Latn', '__label__pap_Latn', '__label__bem_Latn', '__label__mal_Mlym', '__label__kir_Cyrl', '__label__hye_Armn', '__label__smo_Latn', '__label__sin_Sinh', '__label__fij_Latn', '__label__kan_Knda', '__label__pan_Guru', '__label__hau_Latn', '__label__epo_Latn', '__label__gaz_Latn', '__label__tir_Ethi', '__label__bos_Latn', '__label__srp_Cyrl', '__label__hat_Latn', '__label__pag_Latn', '__label__lua_Latn', '__label__war_Latn', '__label__tel_Telu', '__label__tat_Cyrl', '__label__sag_Latn', '__label__lug_Latn', '__label__tum_Latn', '__label__swh_Latn', '__label__umb_Latn', '__label__som_Latn', '__label__gle_Latn', '__label__kng_Latn', '__label__mos_Latn', '__label__lus_Latn', '__label__khk_Cyrl', '__label__asm_Beng', '__label__tuk_Latn', '__label__quy_Latn', '__label__ayr_Latn', '__label__luo_Latn', '__label__tgk_Cyrl', '__label__cat_Latn', '__label__ssw_Latn', '__label__nno_Latn', '__label__cym_Latn', '__label__kik_Latn', '__label__kmb_Latn', '__label__ory_Orya', '__label__bel_Cyrl', '__label__bho_Deva', '__label__apc_Arab', '__label__bak_Cyrl', '__label__jav_Latn', '__label__yue_Hani', '__label__pbt_Arab', '__label__khm_Khmr', '__label__npi_Deva', '__label__npi_Latn', '__label__gug_Latn', '__label__uig_Arab', '__label__fur_Latn', '__label__kbp_Latn', '__label__hne_Deva', '__label__kam_Latn', '__label__gla_Latn', '__label__kab_Latn', '__label__arz_Arab', '__label__kaz_Cyrl', '__label__mri_Latn', '__label__lim_Latn', '__label__srd_Latn', '__label__sun_Latn', '__label__plt_Latn', '__label__mni_Beng', '__label__isl_Latn', '__label__vec_Latn', '__label__glg_Latn', '__label__scn_Latn', '__label__fao_Latn', '__label__san_Deva', '__label__ltz_Latn', '__label__cjk_Latn', '__label__ast_Latn', '__label__lmo_Latn', '__label__szl_Latn', '__label__oci_Latn', '__label__fon_Latn', '__label__min_Latn', '__label__wol_Latn', '__label__lij_Latn', '__label__ajp_Arab', '__label__snd_Arab', '__label__dik_Latn', '__label__ary_Arab', '__label__lao_Laoo', '__label__ars_Arab', '__label__bjn_Latn', '__label__shn_Mymr', '__label__crh_Latn', '__label__aeb_Arab', '__label__ace_Latn', '__label__ckb_Arab', '__label__dyu_Latn', '__label__ltg_Latn', '__label__kmr_Latn', '__label__ban_Latn', '__label__mai_Deva', '__label__fuv_Latn', '__label__kac_Latn', '__label__taq_Latn', '__label__bam_Latn', '__label__sat_Olck', '__label__tzm_Tfng', '__label__bug_Latn', '__label__dzo_Tibt', '__label__kas_Deva', '__label__fas_Arab', '__label__nus_Latn', '__label__knc_Latn', '__label__mag_Deva', '__label__taq_Tfng', '__label__kas_Arab', '__label__knc_Arab', '__label__bjn_Arab', '__label__ace_Arab', '__label__kea_Latn', '__label__awa_Deva', '__label__acm_Arab', '__label__bod_Tibt', '__label__sot_Latn', '__label__ydd_Hebr', '__label__azb_Arab']
    # flores_glotlid = ['__label__eng_Latn', '__label__khm_Khmr']
    # tgt = "في فرنسا ، كانت تجربة التصويتtraditionally بسيطة التقنية: يغلق المشاركون أنفسهم في قاعة التصويت ، ويدخلون ورقة مُطبوعة مسبقاً تحمل اسم المرشح المفضل لديهم في حقيبة."
    tgt = "خلال سنوات ١٩٦٠ ، عمل برزينسكي لجون إف. كينيدي كمشير له ، ثم ل администраة ليندون بي. جونسون."
    # tgt = "Wann eng klein Grupp vu Lebewesen (eng klein Population) vun der Haaptpopulation getrennt gëtt, aus där se komm sinn (wie wann se iwwer eng Bergekette oder eng Fluss migréieren, oder wann se op eng neie Insel migréieren, wou se sech net einfach zréckkommen kënnen), dann entdecken se sech oft an enger anerer Umwelt, wéi virdem."
    # tgt_list = tgt_long.split()
    # final_lang_list = []
    # for tgt in tgt_list:
    lang_detect_model = MaskLID("/mnt/gemini/data1/yifengliu/model/masklid/model_v3.bin", languages=flores_glotlid)
    ans = lang_detect_model.predict_codeswitch(tgt, beta = 20 , alpha = 3, max_lambda = 4, min_length = 10, min_prob = 0.90, max_retry=3, alpha_step_increase = 3, beta_step_increase = 5)
    ans = {key.replace("__label__", ""): value for key, value in ans.items()}
    # tgt_lang = "Luxembourgish"
    tgt_lang = "Arabic"
    long_lang_id = lang2long.get(tgt_lang, None)
    if long_lang_id is None:
        raise ValueError(f"Language code {tgt_lang} not found in lang2long.")
    lang_translation = ans.get(long_lang_id, None)
        # print(tgt_lang, long_lang_id, ans, lang_translation)
    if lang_translation is None:
        lang_translation = ""
  
    
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
        
    dataset = load_dataset("/mnt/gemini/data1/yifengliu/qe-lr/data/train/final_llamax_mix-100k.jsonl")
    # srcs = [data['src'] for data in dataset]
    # preds = [data['pred'] for data in dataset]
    # refs = [data['ref'] for data in dataset]
        
    # bleu = BLEU(tokenize='flores200', smooth_method='exp', lowercase=False)
    # bleu_score = bleu.corpus_score(preds, [refs])
    # print(bleu_score)
    import code; code.interact(local=locals())