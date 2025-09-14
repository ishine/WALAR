import json
import random
import pandas as pd
import re
import os
import sacrebleu
import fasttext
import matplotlib.pyplot as plt
import sys
import tqdm
from tqdm import *
sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
from utils import lang_dict, mm_dict, lang_dict, lang2long, long2lang, my_load_dataset
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
            dataset.append(json.loads(line.strip()))
    return dataset

def get_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    hyps = [hyp.strip() for hyp in hyps]
    refs = [ref.strip() for ref in refs]
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores101", force=True).score
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
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen2.5-3B-Instruct-En-Zh-1M/global_step160_hf/eng-zho_simpl.txt"
    dataset = []
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Seq-Align-Rule-Detect-MetricX-Qwen3-4B-en-cs-1M-bsz128/global_step400_hf/eng-ben.txt"
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B/eng-tgl.txt"
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B/eng-jav.txt"
    
    # with open(path, 'r') as f:
    #     lines = f.readlines()
    #     # import code; code.interact(local=locals())
    #     for line in lines:
    #         try:
    #             dataset.append(json.loads(line))
    #         except:
    #             break
    # srcs = [data['src'] for data in dataset]
    # hyps = [data['pred'] for data in dataset]
    # refs = [data['ref'] for data in dataset]
    # srcs, hyps, refs = srcs[200:300], hyps[200:300], refs[200:300]
    # # hyps, refs = 
    # comet_scores = calculate_comet_score(srcs, refs, hyps, model_path=xcomet_path)['scores']
    # chrf = CHRF()
    
    # bleu_scores = [chrf.sentence_score(hyp, [ref]).score for hyp, ref in zip(hyps, refs)]
    # for i, (x, y) in enumerate(zip(bleu_scores, comet_scores)):
    #     plt.annotate(
    #         str(i),           # label with index
    #         (x, y),           # position of the point
    #         textcoords="offset points",
    #         xytext=(5, 5),    # offset position of label
    #         ha='center',
    #         fontsize=8
    #     )
    
    # plt.scatter(bleu_scores, comet_scores, alpha=0.7, s=60)

    # # Add labels and title
    # plt.xlabel("BLEU Score")
    # plt.ylabel("COMET Score")
    # plt.title("BLEU vs COMET Scores")

    # # Optional: add grid
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.savefig(f"/mnt/gemini/data1/yifengliu/qe-lr/output/temp3.png")
    # plt.show()
    tokenizer = AutoTokenizer.from_pretrained("/mnt/gemini/data1/yifengliu/model/Qwen3-4B")

    # langs = ["ltz", "mkd", "pol", "srp", "slk", "slv", "ben", "guj", "hin", "mar", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav", "tur", "tam", "fin"]
    langs = ["hin"]
    # langs = ["ben"]
    ratio_list = []
    # langs = ["hin"]
    for i in tqdm(range(len(langs))):
        lang = langs[i]
        full_language = lang_dict.get(lang, None)
        if full_language is None:
            raise ValueError(f"Language code {lang} not found in lang_dict.")
        # path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{lang}.devtest"
        # path = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-hin2-1m.jsonl"
        path1 = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B/eng-{lang}.txt"
        path2 = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Seq-Rule-Detect-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step280_hf/eng-{lang}.txt"

        flores_dataset1 = my_load_dataset(path1)
        # flores_dataset = flores_dataset[:1]
        flores_dataset2 = my_load_dataset(path2)
        flores_dataset1 = [data['pred'].replace("\n", "") for data in flores_dataset1]
        flores_dataset2 = [data['pred'].replace("\n", "") for data in flores_dataset2]
        
        # GlotLID has more than 2000 labels, here we limit the GlotLID to the 200 languages available in flores
        flores_glotlid = ['__label__eng_Latn', '__label__arb_Arab', '__label__rus_Cyrl', '__label__por_Latn', '__label__pol_Latn', '__label__ekk_Latn', '__label__ell_Grek', '__label__slk_Latn', '__label__slv_Latn', '__label__nld_Latn', '__label__lvs_Latn', '__label__hun_Latn', '__label__dan_Latn', '__label__swe_Latn', '__label__lit_Latn', '__label__fin_Latn', '__label__mlt_Latn', '__label__cmn_Hani', '__label__nob_Latn', '__label__kor_Hang', '__label__ind_Latn', '__label__uzn_Latn', '__label__fil_Latn', '__label__ukr_Cyrl', '__label__hin_Deva', '__label__hin_Latn', '__label__afr_Latn', '__label__mar_Deva', '__label__ceb_Latn', '__label__ilo_Latn', '__label__zul_Latn', '__label__heb_Hebr', '__label__xho_Latn', '__label__vie_Latn', '__label__jpn_Jpan', '__label__guj_Gujr', '__label__hrv_Latn', '__label__tur_Latn', '__label__nya_Latn', '__label__tsn_Latn', '__label__sna_Latn', '__label__tso_Latn', '__label__tha_Thai', '__label__spa_Latn', '__label__deu_Latn', '__label__eus_Latn', '__label__bul_Cyrl', '__label__amh_Ethi', '__label__fra_Latn', '__label__ewe_Latn', '__label__mkd_Cyrl', '__label__nso_Latn', '__label__tam_Taml', '__label__lin_Latn', '__label__twi_Latn', '__label__yor_Latn', '__label__als_Latn', '__label__ibo_Latn', '__label__ben_Beng', '__label__ita_Latn', '__label__tpi_Latn', '__label__azj_Latn', '__label__run_Latn', '__label__mya_Mymr', '__label__kin_Latn', '__label__ron_Latn', '__label__ces_Latn', '__label__kat_Geor', '__label__urd_Arab', '__label__zsm_Latn', '__label__pap_Latn', '__label__bem_Latn', '__label__mal_Mlym', '__label__kir_Cyrl', '__label__hye_Armn', '__label__smo_Latn', '__label__sin_Sinh', '__label__fij_Latn', '__label__kan_Knda', '__label__pan_Guru', '__label__hau_Latn', '__label__epo_Latn', '__label__gaz_Latn', '__label__tir_Ethi', '__label__bos_Latn', '__label__srp_Cyrl', '__label__hat_Latn', '__label__pag_Latn', '__label__lua_Latn', '__label__war_Latn', '__label__tel_Telu', '__label__tat_Cyrl', '__label__sag_Latn', '__label__lug_Latn', '__label__tum_Latn', '__label__swh_Latn', '__label__umb_Latn', '__label__som_Latn', '__label__gle_Latn', '__label__kng_Latn', '__label__mos_Latn', '__label__lus_Latn', '__label__khk_Cyrl', '__label__asm_Beng', '__label__tuk_Latn', '__label__quy_Latn', '__label__ayr_Latn', '__label__luo_Latn', '__label__tgk_Cyrl', '__label__cat_Latn', '__label__ssw_Latn', '__label__nno_Latn', '__label__cym_Latn', '__label__kik_Latn', '__label__kmb_Latn', '__label__ory_Orya', '__label__bel_Cyrl', '__label__bho_Deva', '__label__apc_Arab', '__label__bak_Cyrl', '__label__jav_Latn', '__label__yue_Hani', '__label__pbt_Arab', '__label__khm_Khmr', '__label__npi_Deva', '__label__npi_Latn', '__label__gug_Latn', '__label__uig_Arab', '__label__fur_Latn', '__label__kbp_Latn', '__label__hne_Deva', '__label__kam_Latn', '__label__gla_Latn', '__label__kab_Latn', '__label__arz_Arab', '__label__kaz_Cyrl', '__label__mri_Latn', '__label__lim_Latn', '__label__srd_Latn', '__label__sun_Latn', '__label__plt_Latn', '__label__mni_Beng', '__label__isl_Latn', '__label__vec_Latn', '__label__glg_Latn', '__label__scn_Latn', '__label__fao_Latn', '__label__san_Deva', '__label__ltz_Latn', '__label__cjk_Latn', '__label__ast_Latn', '__label__lmo_Latn', '__label__szl_Latn', '__label__oci_Latn', '__label__fon_Latn', '__label__min_Latn', '__label__wol_Latn', '__label__lij_Latn', '__label__ajp_Arab', '__label__snd_Arab', '__label__dik_Latn', '__label__ary_Arab', '__label__lao_Laoo', '__label__ars_Arab', '__label__bjn_Latn', '__label__shn_Mymr', '__label__crh_Latn', '__label__aeb_Arab', '__label__ace_Latn', '__label__ckb_Arab', '__label__dyu_Latn', '__label__ltg_Latn', '__label__kmr_Latn', '__label__ban_Latn', '__label__mai_Deva', '__label__fuv_Latn', '__label__kac_Latn', '__label__taq_Latn', '__label__bam_Latn', '__label__sat_Olck', '__label__tzm_Tfng', '__label__bug_Latn', '__label__dzo_Tibt', '__label__kas_Deva', '__label__fas_Arab', '__label__nus_Latn', '__label__knc_Latn', '__label__mag_Deva', '__label__taq_Tfng', '__label__kas_Arab', '__label__knc_Arab', '__label__bjn_Arab', '__label__ace_Arab', '__label__kea_Latn', '__label__awa_Deva', '__label__acm_Arab', '__label__bod_Tibt', '__label__sot_Latn', '__label__ydd_Hebr', '__label__azb_Arab']
        # flores_glotlid = ['__label__eng_Latn', '__label__deu_Latn', '__label__isl_Latn', '__label__ltz_Latn', '__label__bel_Cyrl', '__label__ces_Latn', '__label__mkd_Cyrl', '__label__pol_Latn', '__label__srp_Cyrl', '__label__slk_Latn', '__label__slv_Latn', '__label__ukr_Cyrl', '__label__ben_Beng', '__label__guj_Gujr', '__label__hin_Deva', '__label__mar_Deva', '__label__npi_Deva', '__label__pan_Guru', '__label__urd_Arab', '__label__hye_Armn', '__label__ell_Grek', '__label__lvs_Latn', '__label__lit_Latn', '__label__fas_Arab', '__label__cym_Latn', '__label__ceb_Latn', '__label__jav_Latn', '__label__arb_Arab', '__label__azj_Latn', '__label__kaz_Cyrl', '__label__tur_Latn', '__label__uzn_Latn', '__label__kan_Knda', '__label__mal_Mlym', '__label__tam_Taml', '__label__tel_Telu', '__label__mya_Mymr', '__label__ekk_Latn', '__label__fin_Latn', '__label__hun_Latn', '__label__kat_Geor', '__label__heb_Hebr', '__label__khm_Khmr', '__label__kor_Hang', '__label__lao_Laoo', '__label__fil_Latn']
        # flores_glotlid = [f"__label__{lang2long[full_language]}", "__label__eng_Latn"]
        model_path = "/mnt/gemini/data1/yifengliu/model/masklid/model_v3.bin"
        import masklid
        from masklid import MaskLID
        # import masklid2
        # from masklid2 import MaskLID
        masklid_model = MaskLID(model_path, languages=flores_glotlid)
    
        answers = []
        # flores_dataset1 = ["कार्यस्थल में समानता (harmony) महत्वपूर्ण है, जहां समूह के सहयोग (group effort) को सम्मानित किया जाता है, न कि व्यक्तिगत सफलता (individual accomplishments) को।"]
        # ratio_list1 = get_ratio_list(masklid_model, flores_dataset1, full_language, tokenizer)
        ratio_list1 = get_ratio_list(masklid_model, flores_dataset2, full_language, tokenizer)
        # ratio_list1 = [0]*len(ratio_list2)
        # ratio_list2 = [0]*len(ratio_list1)
        # ratio_list.extend([ratio2 - ratio1 for ratio1, ratio2 in zip(ratio_list1, ratio_list2)])
        # ratio_list.extend([ratio1 - ratio2 for ratio1, ratio2 in zip(ratio_list1, ratio_list2)])
        # import code; code.interact(local=locals())

    for i in tqdm(range(len(langs))):
        lang = langs[i]
        full_language = lang_dict.get(lang, None)
        if full_language is None:
            raise ValueError(f"Language code {lang} not found in lang_dict.")
        # path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{lang}.devtest"
        # path = f"/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-hin2-1m.jsonl"
        path1 = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B/eng-{lang}.txt"
        path2 = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Seq-Rule-Detect-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step280_hf/eng-{lang}.txt"

        flores_dataset1 = my_load_dataset(path1)
        # flores_dataset = flores_dataset[:1]
        flores_dataset2 = my_load_dataset(path2)
        flores_dataset1 = [data['pred'].replace("\n", "") for data in flores_dataset1]
        flores_dataset2 = [data['pred'].replace("\n", "") for data in flores_dataset2]
        
        # GlotLID has more than 2000 labels, here we limit the GlotLID to the 200 languages available in flores
        flores_glotlid = ['__label__eng_Latn', '__label__arb_Arab', '__label__rus_Cyrl', '__label__por_Latn', '__label__pol_Latn', '__label__ekk_Latn', '__label__ell_Grek', '__label__slk_Latn', '__label__slv_Latn', '__label__nld_Latn', '__label__lvs_Latn', '__label__hun_Latn', '__label__dan_Latn', '__label__swe_Latn', '__label__lit_Latn', '__label__fin_Latn', '__label__mlt_Latn', '__label__cmn_Hani', '__label__nob_Latn', '__label__kor_Hang', '__label__ind_Latn', '__label__uzn_Latn', '__label__fil_Latn', '__label__ukr_Cyrl', '__label__hin_Deva', '__label__hin_Latn', '__label__afr_Latn', '__label__mar_Deva', '__label__ceb_Latn', '__label__ilo_Latn', '__label__zul_Latn', '__label__heb_Hebr', '__label__xho_Latn', '__label__vie_Latn', '__label__jpn_Jpan', '__label__guj_Gujr', '__label__hrv_Latn', '__label__tur_Latn', '__label__nya_Latn', '__label__tsn_Latn', '__label__sna_Latn', '__label__tso_Latn', '__label__tha_Thai', '__label__spa_Latn', '__label__deu_Latn', '__label__eus_Latn', '__label__bul_Cyrl', '__label__amh_Ethi', '__label__fra_Latn', '__label__ewe_Latn', '__label__mkd_Cyrl', '__label__nso_Latn', '__label__tam_Taml', '__label__lin_Latn', '__label__twi_Latn', '__label__yor_Latn', '__label__als_Latn', '__label__ibo_Latn', '__label__ben_Beng', '__label__ita_Latn', '__label__tpi_Latn', '__label__azj_Latn', '__label__run_Latn', '__label__mya_Mymr', '__label__kin_Latn', '__label__ron_Latn', '__label__ces_Latn', '__label__kat_Geor', '__label__urd_Arab', '__label__zsm_Latn', '__label__pap_Latn', '__label__bem_Latn', '__label__mal_Mlym', '__label__kir_Cyrl', '__label__hye_Armn', '__label__smo_Latn', '__label__sin_Sinh', '__label__fij_Latn', '__label__kan_Knda', '__label__pan_Guru', '__label__hau_Latn', '__label__epo_Latn', '__label__gaz_Latn', '__label__tir_Ethi', '__label__bos_Latn', '__label__srp_Cyrl', '__label__hat_Latn', '__label__pag_Latn', '__label__lua_Latn', '__label__war_Latn', '__label__tel_Telu', '__label__tat_Cyrl', '__label__sag_Latn', '__label__lug_Latn', '__label__tum_Latn', '__label__swh_Latn', '__label__umb_Latn', '__label__som_Latn', '__label__gle_Latn', '__label__kng_Latn', '__label__mos_Latn', '__label__lus_Latn', '__label__khk_Cyrl', '__label__asm_Beng', '__label__tuk_Latn', '__label__quy_Latn', '__label__ayr_Latn', '__label__luo_Latn', '__label__tgk_Cyrl', '__label__cat_Latn', '__label__ssw_Latn', '__label__nno_Latn', '__label__cym_Latn', '__label__kik_Latn', '__label__kmb_Latn', '__label__ory_Orya', '__label__bel_Cyrl', '__label__bho_Deva', '__label__apc_Arab', '__label__bak_Cyrl', '__label__jav_Latn', '__label__yue_Hani', '__label__pbt_Arab', '__label__khm_Khmr', '__label__npi_Deva', '__label__npi_Latn', '__label__gug_Latn', '__label__uig_Arab', '__label__fur_Latn', '__label__kbp_Latn', '__label__hne_Deva', '__label__kam_Latn', '__label__gla_Latn', '__label__kab_Latn', '__label__arz_Arab', '__label__kaz_Cyrl', '__label__mri_Latn', '__label__lim_Latn', '__label__srd_Latn', '__label__sun_Latn', '__label__plt_Latn', '__label__mni_Beng', '__label__isl_Latn', '__label__vec_Latn', '__label__glg_Latn', '__label__scn_Latn', '__label__fao_Latn', '__label__san_Deva', '__label__ltz_Latn', '__label__cjk_Latn', '__label__ast_Latn', '__label__lmo_Latn', '__label__szl_Latn', '__label__oci_Latn', '__label__fon_Latn', '__label__min_Latn', '__label__wol_Latn', '__label__lij_Latn', '__label__ajp_Arab', '__label__snd_Arab', '__label__dik_Latn', '__label__ary_Arab', '__label__lao_Laoo', '__label__ars_Arab', '__label__bjn_Latn', '__label__shn_Mymr', '__label__crh_Latn', '__label__aeb_Arab', '__label__ace_Latn', '__label__ckb_Arab', '__label__dyu_Latn', '__label__ltg_Latn', '__label__kmr_Latn', '__label__ban_Latn', '__label__mai_Deva', '__label__fuv_Latn', '__label__kac_Latn', '__label__taq_Latn', '__label__bam_Latn', '__label__sat_Olck', '__label__tzm_Tfng', '__label__bug_Latn', '__label__dzo_Tibt', '__label__kas_Deva', '__label__fas_Arab', '__label__nus_Latn', '__label__knc_Latn', '__label__mag_Deva', '__label__taq_Tfng', '__label__kas_Arab', '__label__knc_Arab', '__label__bjn_Arab', '__label__ace_Arab', '__label__kea_Latn', '__label__awa_Deva', '__label__acm_Arab', '__label__bod_Tibt', '__label__sot_Latn', '__label__ydd_Hebr', '__label__azb_Arab']
        # flores_glotlid = ['__label__eng_Latn', '__label__deu_Latn', '__label__isl_Latn', '__label__ltz_Latn', '__label__bel_Cyrl', '__label__ces_Latn', '__label__mkd_Cyrl', '__label__pol_Latn', '__label__srp_Cyrl', '__label__slk_Latn', '__label__slv_Latn', '__label__ukr_Cyrl', '__label__ben_Beng', '__label__guj_Gujr', '__label__hin_Deva', '__label__mar_Deva', '__label__npi_Deva', '__label__pan_Guru', '__label__urd_Arab', '__label__hye_Armn', '__label__ell_Grek', '__label__lvs_Latn', '__label__lit_Latn', '__label__fas_Arab', '__label__cym_Latn', '__label__ceb_Latn', '__label__jav_Latn', '__label__arb_Arab', '__label__azj_Latn', '__label__kaz_Cyrl', '__label__tur_Latn', '__label__uzn_Latn', '__label__kan_Knda', '__label__mal_Mlym', '__label__tam_Taml', '__label__tel_Telu', '__label__mya_Mymr', '__label__ekk_Latn', '__label__fin_Latn', '__label__hun_Latn', '__label__kat_Geor', '__label__heb_Hebr', '__label__khm_Khmr', '__label__kor_Hang', '__label__lao_Laoo', '__label__fil_Latn']
        # flores_glotlid = [f"__label__{lang2long[full_language]}", "__label__eng_Latn"]
        model_path = "/mnt/gemini/data1/yifengliu/model/masklid/model_v3.bin"
        # import masklid
        # from masklid import MaskLID
        import masklid2
        from masklid2 import MaskLID
        masklid_model = MaskLID(model_path, languages=flores_glotlid)
    
        answers = []
        # flores_dataset1 = ["कार्यस्थल में समानता (harmony) महत्वपूर्ण है, जहां समूह के सहयोग (group effort) को सम्मानित किया जाता है, न कि व्यक्तिगत सफलता (individual accomplishments) को।"]
        # ratio_list1 = get_ratio_list(masklid_model, flores_dataset1, full_language, tokenizer)
        ratio_list2 = get_ratio_list(masklid_model, flores_dataset2, full_language, tokenizer)
        # ratio_list1 = [0]*len(ratio_list2)
        # ratio_list2 = [0]*len(ratio_list1)
        # ratio_list.extend([ratio2 - ratio1 for ratio1, ratio2 in zip(ratio_list1, ratio_list2)])
        # ratio_list.extend([ratio1 - ratio2 for ratio1, ratio2 in zip(ratio_list1, ratio_list2)])
    
    lst = list(range(0, 100, 1))
    plt.hist(ratio_list, bins=[temp / 100 for temp in lst], edgecolor='black')
    plt.xlabel('Ratio Range')
    plt.ylabel('Count')
    plt.title('Token Length Ratio Distribution (Detected/Original)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"/mnt/gemini/data1/yifengliu/qe-lr/output/ratio_gap2.png")
    # print(len([ans for ans in answers if ans.get("ekk_Latn", None) is None]))
    
            
    # tokenizer = AutoTokenizer.from_pretrained("/mnt/gemini/data1/yifengliu/model/Qwen3-4B")
    # srcs_tokens_length = tokenizer(srcs)['input_ids']
    # hyps_tokens_length = tokenizer(hyps)['input_ids']
    # import code; code.interact(local=locals())
    # ratio_list = [len(hyp) / len(src) for src, hyp in zip(srcs_tokens_length, hyps_tokens_length)]
    
    # plt.hist(ratio_list, bins=[i/10 for i in range(41)], edgecolor='black')

    # plt.xlabel('Value Range')
    # plt.ylabel('Count')
    # plt.title('F1 Score Distribution')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # # plt.show()
    # plt.savefig(f"/mnt/gemini/data1/yifengliu/qe-lr/output/temp.png")
    # print(f"Align Score: {align_score_list}")

    
    # lang_detect_model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
    # hyps2 = [tgt.replace("\n", "") for tgt in hyps]
    # lang_info = lang_detect_model.predict(hyps2)
    
    # cnt = 0
    # for idx, (lang, hyp) in enumerate(zip(lang_info[0], hyps)):
    #     lang_code = lang[0].replace("__label__", "")
    #     if lang_code != "bn":
    #         print(f"{idx}: {hyp}")
    #         cnt += 1
    # print("Done!")
    
    # for idx, (lang, src, hyp, ref) in enumerate(zip(lang_info[0], srcs, hyps, refs)):
        # lang_code = lang[0].replace("__label__", "")
        # if lang_code != "ja":
            # print(f"{idx}: {hyp}")
            # print("============================")
    
    
    # dataset[0]['label']=True
    # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/est.jsonl"
    # with open(save_path, 'w') as f:
    #     for data in dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # score = get_spBLEU(hyps, refs)
    # tgt_list = ["ara, bel, ben, deu, fin, glg, eng-hin"]
    # xcomet_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen2.5-3B-Instruct/eng-tur.txt"
    # with open(path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines[:-2]:
    #         data = json.loads(line.strip())
    #         dataset.append(data)
    # model = load_from_checkpoint(xcomet_path)
    
    # inputs = [{"src": data['src'].strip(), "mt": data['pred'].strip(), "ref": data['ref'].strip()} for data in dataset]
    # output = model.predict(inputs)
    
    # scores, mean_score = output.scores, output.system_score
    
    
    
    
    import code; code.interact(local=locals())