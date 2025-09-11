import dataclasses
import argparse
import os
import sys
import re
import jieba
# import hanlp
# import hanlp_restful
# from hanlp_restful import HanLPClient

sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
import masklid
from masklid import MaskLID
from utils import lang2long, long2lang
import models
# from sentence_transformers import SentenceTransformer
from typing import Any, List, Tuple, Union, Optional
from tqdm import *

import torch
import transformers
import datasets
import uvicorn
import fasttext
import sacrebleu
import itertools
from transformers import AutoTokenizer, AutoModel
# from simalign import SentenceAligner
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import DataCollatorWithPadding

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

lang_dict = {
  "af": "Afrikaans",
  "am": "Amharic",
  "ar": "Arabic",
  "hy": "Armenian",
  "as": "Assamese",
  "ast": "Asturian",
  "az": "Azerbaijani",
  "be": "Belarusian",
  "bn": "Bengali",
  "bs": "Bosnian",
  "bg": "Bulgarian",
  "my": "Burmese",
  "ca": "Catalan",
  "ceb": "Cebuano",
  "zh": "Chinese",
  "zho": "Chinese",
  "hr": "Croatian",
  "cs": "Czech",
  "da": "Danish",
  "nl": "Dutch",
  "en": "English",
  "et": "Estonian",
  "tl": "Filipino",
  "fi": "Finnish",
  "fr": "French",
  "ff": "Fulah",
  "gl": "Galician",
  "lg": "Ganda",
  "ka": "Georgian",
  "de": "German",
  "el": "Greek",
  "gu": "Gujarati",
  "ha": "Hausa",
  "he": "Hebrew",
  "hi": "Hindi",
  "hu": "Hungarian",
  "is": "Icelandic",
  "ig": "Igbo",
  "id": "Indonesian",
  "ga": "Irish",
  "it": "Italian",
  "ja": "Japanese",
  "jv": "Javanese",
  "kea": "Kabuverdianu",
  "kam": "Kamba",
  "kn": "Kannada",
  "kk": "Kazakh",
  "km": "Khmer",
  "ko": "Korean",
  "ky": "Kyrgyz",
  "lo": "Lao",
  "lv": "Latvian",
  "ln": "Lingala",
  "lt": "Lithuanian",
  "luo": "Luo",
  "lb": "Luxembourgish",
  "mk": "Macedonian",
  "ms": "Malay",
  "ml": "Malayalam",
  "mt": "Maltese",
  "mi": "Maori",
  "mr": "Marathi",
  "mn": "Mongolian",
  "ne": "Nepali",
  "ns": "Northern Sotho",
  "no": "Norwegian",
  "ny": "Nyanja",
  "oc": "Occitan",
  "or": "Oriya",
  "om": "Oromo",
  "ps": "Pashto",
  "fa": "Persian",
  "pl": "Polish",
  "pt": "Portuguese",
  "pa": "Punjabi",
  "ro": "Romanian",
  "ru": "Russian",
  "sr": "Serbian",
  "sn": "Shona",
  "sd": "Sindhi",
  "sk": "Slovak",
  "sl": "Slovenian",
  "so": "Somali",
  "ku": "Sorani Kurdish",
  "es": "Spanish",
  "sw": "Swahili",
  "sv": "Swedish",
  "tg": "Tajik",
  "ta": "Tamil",
  "te": "Telugu",
  "th": "Thai",
  "tr": "Turkish",
  "uk": "Ukrainian",
  "umb": "Umbundu",
  "ur": "Urdu",
  "uz": "Uzbek",
  "vi": "Vietnamese",
  "cy": "Welsh",
  "wo": "Wolof",
  "xh": "Xhosa",
  "yo": "Yoruba",
  "zu": "Zulu",
}

@dataclasses.dataclass
class Arguments:
  model_name: str = dataclasses.field(
        default="metricX",
        metadata={
            "help": "The name of the model to use. Supported models: 'metricX', etc."
        }
    )

  value_head_prefix: str = dataclasses.field(
      default="score",
      metadata={"help": "Prefix for the value head"}
  )

  max_len: int = dataclasses.field(
      default=2048,
      metadata={"help": "The maximum sequence length for the model input"}
  )

  port: int = dataclasses.field(
      default=5000,
      metadata={"help": "Port number for the server"}
  )

  host: str = dataclasses.field(
      default="0.0.0.0",
      metadata={"help": "IP address for the server"}
  )

  base_model: str = dataclasses.field(
      default="Qwen2.5-3B-Instruct",
      metadata={"help": "Base model name or path"}
  )

  lang_detect: bool = dataclasses.field(
      default=False,
      metadata={"help": "Enable language detection"}
  )

  rule: bool = dataclasses.field(
      default=False,
      metadata={"help": "Rule to use \\n as a reward or not"}
  )
  
  truncate: bool = dataclasses.field(
    default=False,
    metadata={"help": "Truncate the reward or not"}
  )
  
  bleu: bool = dataclasses.field(
    default=False,
    metadata={"help": "Enable BLEU metric"}
  )
  
  align: bool = dataclasses.field(
    default=False,
    metadata={"help": "Enable alignment model"}
  )

  load_in_4bit: bool = dataclasses.field(
      default=False,
      metadata={"help": "Load model in 4-bit precision"}
  )

  bf16: bool = dataclasses.field(
      default=False,
      metadata={"help": "Enable bfloat16 (bf16) precision"}
  )

  disable_fast_tokenizer: bool = dataclasses.field(
      default=False,
      metadata={"help": "Disable the use of fast tokenizer"}
  )

  packing_samples: bool = dataclasses.field(
      default=False,
      metadata={"help": "Enable packing of input samples"}
  )

  batch_size: Optional[int] = dataclasses.field(
      default=None,
      metadata={"help": "Batch size for prediction or inference"}
  )

  use_ms: bool = dataclasses.field(
      default=False,
      metadata={"help": "Enable ModelScope usage"}
  )
  
  src: str = dataclasses.field(
      default="en",
      metadata={"help": "Source language for translation"}
  )
  
  tgt: str = dataclasses.field(
      default="zh",
      metadata={"help": "Target language for translation"}
  )

def get_dataset(
    ds: List[dict], model_name: str, tokenizer, max_input_length: int, device, is_qe: bool
):
  """Gets the test dataset for prediction.

  If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
  If it is false, there must be "hypothesis" and "reference" fields.

  Args:
    input_file: The path to the jsonl input file.
    tokenizer: The tokenizer to use.
    max_input_length: The maximum input sequence length.
    device: The ID of the device to put the PyTorch tensors on.
    is_qe: Indicates whether the metric is a QE metric or not.

  Returns:
    The dataset.
  """

  def _make_input(example):
    if 'metricX' in model_name:
      if is_qe:
        example["input"] = (
            "source: "
            + example["source"]
            + " candidate: "
            + example["hypothesis"]
        )
      else:
        example["input"] = (
            "source: "
            + example["source"]
            + " candidate: "
            + example["hypothesis"]
            + " reference: "
            + example["reference"]
        )
    elif 'Comet' in model_name:
      src = example.pop("source", "")
      mt = example.pop("hypothesis", "")
      if src == "" or mt == "":
        raise ValueError(
            "Input data must have 'source' and 'hypothesis' fields for Comet models."
        )
      if is_qe:
        example["input"] = {
          "src": src, 
          "mt": mt
        }
      else:
        ref = example.pop("reference", "")
        example["input"] = {
          "src": src, 
          "mt": mt, 
          "ref": ref
        }
    else:
      raise ValueError("Unsupported model name in Dataset Processing: {}".format(model_name))
    return example

  def _tokenize(example):
    return tokenizer(
        example["input"],
        max_length=max_input_length,
        truncation=True,
        padding=True,
    )

  def _remove_eos(example):
    example["input_ids"] = example["input_ids"][:-1]
    example["attention_mask"] = example["attention_mask"][:-1]
    return example
  
  if "metricX" in model_name:
    ds = ds.map(_make_input)
    ds = ds.map(_tokenize)
    ds = ds.map(_remove_eos)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        device=device,
        output_all_columns=True,
    )
  elif "Comet" in model_name:
    ds = ds.map(_make_input)

  return ds

def get_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    hyps = [hyp.strip() for hyp in hyps]
    refs = [ref.strip() for ref in refs]
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="spm", force=True).score
    return result

def simple_align(srcs, tgts, aligner):
  align_score_list = []
  for i in tqdm(range(len(srcs))):
    src_sentence = srcs[i]
    trg_sentence = tgts[i]

    # The output is a dictionary with different matching methods.
    # Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
    alignments = aligner.get_word_aligns(src_sentence, trg_sentence)
    # import code; code.interact(local=locals())
    src_words = set({t[0]: t for t in sorted(alignments['mwmf'])}.values())
    tgt_words = set({t[1]: t for t in sorted(alignments['mwmf'])}.values())

    precision = min(len(tgt_words) / len(trg_sentence), 1)
    recall = min(len(src_words) / len(src_sentence), 1)
    f1 = 2 * precision * recall / (precision + recall)
    align_score_list.append(f1)
  return align_score_list

def align_score(srcs, tgts, model, tokenizer, batch_size=16):
    align_score_list = []
    align_layer = 24
    threshold = 1e-3

    # 将模型移动到GPU（如果可用）
    device = model.device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    # 预计算所有tokenizations和映射
    tokenized_data = []
    
    # 预处理所有数据
    for i in tqdm(range(len(srcs))):
        sent_src, sent_tgt = srcs[i], tgts[i]
        
        # 同时处理源语言和目标语言
        token_src = [tokenizer.tokenize(word) for word in sent_src]
        token_tgt = [tokenizer.tokenize(word) for word in sent_tgt]
        
        # 转换为ID
        wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
        wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        
        # 准备模型输入
        ids_src = tokenizer.prepare_for_model(
            list(itertools.chain(*wid_src)), 
            return_tensors='pt', 
            truncation=True,
            max_length=tokenizer.model_max_length
        )['input_ids']
        
        ids_tgt = tokenizer.prepare_for_model(
            list(itertools.chain(*wid_tgt)), 
            return_tensors='pt', 
            truncation=True,
            max_length=tokenizer.model_max_length
        )['input_ids']
        
        # 创建子词到单词的映射
        sub2word_map_src = []
        for idx, word_list in enumerate(token_src):
            sub2word_map_src.extend([idx] * len(word_list))
            
        sub2word_map_tgt = []
        for idx, word_list in enumerate(token_tgt):
            sub2word_map_tgt.extend([idx] * len(word_list))
        
        tokenized_data.append({
            'input_ids_src': ids_src.squeeze(),
            'input_ids_tgt': ids_tgt.squeeze(),
            'sub2word_src': sub2word_map_src,
            'sub2word_tgt': sub2word_map_tgt,
            'src_len': len(sent_src),
            'tgt_len': len(sent_tgt)
        })
    
    # 批量处理
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_data), batch_size)):
            batch = tokenized_data[i:i+batch_size]
            
            # 准备批次输入
            src_batch = [item['input_ids_src'] for item in batch]
            tgt_batch = [item['input_ids_tgt'] for item in batch]
            
            # 获取实际长度（排除填充）
            src_lengths = [len(ids) for ids in src_batch]
            tgt_lengths = [len(ids) for ids in tgt_batch]
            
            # 填充批次并移动到设备
            src_tensors = torch.nn.utils.rnn.pad_sequence(
                src_batch, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)
            
            tgt_tensors = torch.nn.utils.rnn.pad_sequence(
                tgt_batch, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)
            
            # 创建注意力掩码
            src_mask = (src_tensors != tokenizer.pad_token_id).to(device)
            tgt_mask = (tgt_tensors != tokenizer.pad_token_id).to(device)
            
            # 获取模型输出
            out_src = model(src_tensors, attention_mask=src_mask, output_hidden_states=True)[2][align_layer]
            out_tgt = model(tgt_tensors, attention_mask=tgt_mask, output_hidden_states=True)[2][align_layer]
            
            # 处理批次中的每个句子
            for j in range(len(batch)):
                item = batch[j]
                
                # 移除特殊标记 ([CLS] 和 [SEP])
                src_start, src_end = 1, src_lengths[j] - 1
                tgt_start, tgt_end = 1, tgt_lengths[j] - 1
                
                valid_src = out_src[j, src_start:src_end]  # 移除 [CLS] 和 [SEP]
                valid_tgt = out_tgt[j, tgt_start:tgt_end]
                
                # 计算对齐
                dot_prod = torch.matmul(valid_src, valid_tgt.transpose(-1, -2))
                
                # 使用更高效的softmax计算
                softmax_srctgt = torch.softmax(dot_prod, dim=-1)
                softmax_tgtsrc = torch.softmax(dot_prod, dim=-2)
                
                # 创建对齐掩码
                softmax_inter = (softmax_srctgt > threshold) & (softmax_tgtsrc > threshold)
                
                # 转换为词对齐
                align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
                
                # 使用集合推导式提高效率
                align_words = {
                    (item['sub2word_src'][i_sub.item()], item['sub2word_tgt'][j_sub.item()])
                    for i_sub, j_sub in align_subwords
                }
                
                # 计算分数
                src_words = {t[0] for t in align_words}
                tgt_words = {t[1] for t in align_words}
                n_src = len(src_words)
                n_tgt = len(tgt_words)
                
                precision = n_tgt / item['tgt_len'] if n_tgt > 0 else 0
                recall = n_src / item['src_len'] if n_src > 0 else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                align_score_list.append(f1)
    
    return align_score_list

def get_ratio_list(model, flores_dataset, full_language, tokenizer):
    # flores_glotlid = ['__label__eng_Latn', '__label__deu_Latn', '__label__isl_Latn', '__label__ltz_Latn', '__label__bel_Cyrl', '__label__ces_Latn', '__label__mkd_Cyrl', '__label__pol_Latn', '__label__srp_Cyrl', '__label__slk_Latn', '__label__slv_Latn', '__label__ukr_Cyrl', '__label__ben_Beng', '__label__guj_Gujr', '__label__hin_Deva', '__label__mar_Deva', '__label__npi_Deva', '__label__pan_Guru', '__label__urd_Arab', '__label__hye_Armn', '__label__ell_Grek', '__label__lvs_Latn', '__label__lit_Latn', '__label__fas_Arab', '__label__cym_Latn', '__label__ceb_Latn', '__label__jav_Latn', '__label__arb_Arab', '__label__azj_Latn', '__label__kaz_Cyrl', '__label__tur_Latn', '__label__uzn_Latn', '__label__kan_Knda', '__label__mal_Mlym', '__label__tam_Taml', '__label__tel_Telu', '__label__mya_Mymr', '__label__ekk_Latn', '__label__fin_Latn', '__label__hun_Latn', '__label__kat_Geor', '__label__heb_Hebr', '__label__khm_Khmr', '__label__kor_Hang', '__label__lao_Laoo', '__label__fil_Latn']
    answers = []
    for i in tqdm(range(len(flores_dataset))):
        text = flores_dataset[i]
        ans = model.predict_codeswitch(text, beta = 20 , alpha = 3, max_lambda = 3, min_length = 10, min_prob = 0.90, max_retry=3, alpha_step_increase = 3, beta_step_increase = 5)
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

class RewardModelProxy:
    def __init__(self, args):
        self.args = args
        self.base_model = args.base_model
        model_path_dict = {
          "Qwen": "/mnt/gemini/data1/yifengliu/model/Qwen3-4B",
          "Llama": "/mnt/gemini/data1/yifengliu/model/Llama-3.2-3B-Instruct",
        }
        if "Qwen" in args.base_model:
          self.base_tokenizer = AutoTokenizer.from_pretrained(model_path_dict["Qwen"])
        elif "Llama" in args.base_model:
          self.base_tokenizer = AutoTokenizer.from_pretrained(model_path_dict["Llama"])
        else:
          raise ValueError(f"Unsupported base model: {args.base_model}")
        if args.lang_detect:
          # flores_glotlid = ['__label__eng_Latn', '__label__deu_Latn', '__label__isl_Latn', '__label__ltz_Latn', '__label__bel_Cyrl', '__label__ces_Latn', '__label__mkd_Cyrl', '__label__pol_Latn', '__label__srp_Cyrl', '__label__slk_Latn', '__label__slv_Latn', '__label__ukr_Cyrl', '__label__ben_Beng', '__label__guj_Gujr', '__label__hin_Deva', '__label__mar_Deva', '__label__npi_Deva', '__label__pan_Guru', '__label__urd_Arab', '__label__hye_Armn', '__label__ell_Grek', '__label__lvs_Latn', '__label__lit_Latn', '__label__fas_Arab', '__label__cym_Latn', '__label__ceb_Latn', '__label__jav_Latn', '__label__arb_Arab', '__label__azj_Latn', '__label__kaz_Cyrl', '__label__tur_Latn', '__label__uzn_Latn', '__label__kan_Knda', '__label__mal_Mlym', '__label__tam_Taml', '__label__tel_Telu', '__label__mya_Mymr', '__label__ekk_Latn', '__label__fin_Latn', '__label__hun_Latn', '__label__kat_Geor', '__label__heb_Hebr', '__label__khm_Khmr', '__label__kor_Hang', '__label__lao_Laoo', '__label__fil_Latn']
          flores_glotlid = ['__label__eng_Latn', '__label__arb_Arab', '__label__rus_Cyrl', '__label__por_Latn', '__label__pol_Latn', '__label__ekk_Latn', '__label__ell_Grek', '__label__slk_Latn', '__label__slv_Latn', '__label__nld_Latn', '__label__lvs_Latn', '__label__hun_Latn', '__label__dan_Latn', '__label__swe_Latn', '__label__lit_Latn', '__label__fin_Latn', '__label__mlt_Latn', '__label__cmn_Hani', '__label__nob_Latn', '__label__kor_Hang', '__label__ind_Latn', '__label__uzn_Latn', '__label__fil_Latn', '__label__ukr_Cyrl', '__label__hin_Deva', '__label__hin_Latn', '__label__afr_Latn', '__label__mar_Deva', '__label__ceb_Latn', '__label__ilo_Latn', '__label__zul_Latn', '__label__heb_Hebr', '__label__xho_Latn', '__label__vie_Latn', '__label__jpn_Jpan', '__label__guj_Gujr', '__label__hrv_Latn', '__label__tur_Latn', '__label__nya_Latn', '__label__tsn_Latn', '__label__sna_Latn', '__label__tso_Latn', '__label__tha_Thai', '__label__spa_Latn', '__label__deu_Latn', '__label__eus_Latn', '__label__bul_Cyrl', '__label__amh_Ethi', '__label__fra_Latn', '__label__ewe_Latn', '__label__mkd_Cyrl', '__label__nso_Latn', '__label__tam_Taml', '__label__lin_Latn', '__label__twi_Latn', '__label__yor_Latn', '__label__als_Latn', '__label__ibo_Latn', '__label__ben_Beng', '__label__ita_Latn', '__label__tpi_Latn', '__label__azj_Latn', '__label__run_Latn', '__label__mya_Mymr', '__label__kin_Latn', '__label__ron_Latn', '__label__ces_Latn', '__label__kat_Geor', '__label__urd_Arab', '__label__zsm_Latn', '__label__pap_Latn', '__label__bem_Latn', '__label__mal_Mlym', '__label__kir_Cyrl', '__label__hye_Armn', '__label__smo_Latn', '__label__sin_Sinh', '__label__fij_Latn', '__label__kan_Knda', '__label__pan_Guru', '__label__hau_Latn', '__label__epo_Latn', '__label__gaz_Latn', '__label__tir_Ethi', '__label__bos_Latn', '__label__srp_Cyrl', '__label__hat_Latn', '__label__pag_Latn', '__label__lua_Latn', '__label__war_Latn', '__label__tel_Telu', '__label__tat_Cyrl', '__label__sag_Latn', '__label__lug_Latn', '__label__tum_Latn', '__label__swh_Latn', '__label__umb_Latn', '__label__som_Latn', '__label__gle_Latn', '__label__kng_Latn', '__label__mos_Latn', '__label__lus_Latn', '__label__khk_Cyrl', '__label__asm_Beng', '__label__tuk_Latn', '__label__quy_Latn', '__label__ayr_Latn', '__label__luo_Latn', '__label__tgk_Cyrl', '__label__cat_Latn', '__label__ssw_Latn', '__label__nno_Latn', '__label__cym_Latn', '__label__kik_Latn', '__label__kmb_Latn', '__label__ory_Orya', '__label__bel_Cyrl', '__label__bho_Deva', '__label__apc_Arab', '__label__bak_Cyrl', '__label__jav_Latn', '__label__yue_Hani', '__label__pbt_Arab', '__label__khm_Khmr', '__label__npi_Deva', '__label__npi_Latn', '__label__gug_Latn', '__label__uig_Arab', '__label__fur_Latn', '__label__kbp_Latn', '__label__hne_Deva', '__label__kam_Latn', '__label__gla_Latn', '__label__kab_Latn', '__label__arz_Arab', '__label__kaz_Cyrl', '__label__mri_Latn', '__label__lim_Latn', '__label__srd_Latn', '__label__sun_Latn', '__label__plt_Latn', '__label__mni_Beng', '__label__isl_Latn', '__label__vec_Latn', '__label__glg_Latn', '__label__scn_Latn', '__label__fao_Latn', '__label__san_Deva', '__label__ltz_Latn', '__label__cjk_Latn', '__label__ast_Latn', '__label__lmo_Latn', '__label__szl_Latn', '__label__oci_Latn', '__label__fon_Latn', '__label__min_Latn', '__label__wol_Latn', '__label__lij_Latn', '__label__ajp_Arab', '__label__snd_Arab', '__label__dik_Latn', '__label__ary_Arab', '__label__lao_Laoo', '__label__ars_Arab', '__label__bjn_Latn', '__label__shn_Mymr', '__label__crh_Latn', '__label__aeb_Arab', '__label__ace_Latn', '__label__ckb_Arab', '__label__dyu_Latn', '__label__ltg_Latn', '__label__kmr_Latn', '__label__ban_Latn', '__label__mai_Deva', '__label__fuv_Latn', '__label__kac_Latn', '__label__taq_Latn', '__label__bam_Latn', '__label__sat_Olck', '__label__tzm_Tfng', '__label__bug_Latn', '__label__dzo_Tibt', '__label__kas_Deva', '__label__fas_Arab', '__label__nus_Latn', '__label__knc_Latn', '__label__mag_Deva', '__label__taq_Tfng', '__label__kas_Arab', '__label__knc_Arab', '__label__bjn_Arab', '__label__ace_Arab', '__label__kea_Latn', '__label__awa_Deva', '__label__acm_Arab', '__label__bod_Tibt', '__label__sot_Latn', '__label__ydd_Hebr', '__label__azb_Arab']
          self.lang_detect_model2 = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
          self.lang_detect_model = MaskLID("/mnt/gemini/data1/yifengliu/model/masklid/model_v3.bin", languages=flores_glotlid)
        if args.align:
          # another potential model: bert-base-multilingual-cased
          model_path = "/mnt/gemini/data1/yifengliu/model/bge-m3"
          # self.align_model = SentenceTransformer(model_path)
          # self.align_model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
          # self.align_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
          self.align_model = AutoModel.from_pretrained(model_path)
          self.align_tokenizer = AutoTokenizer.from_pretrained(model_path)
          self.align_model.to("cuda:0")
          # self.aligner = SentenceAligner(model="bge-m3", token_type="bpe", matching_methods="m", device="cuda:0")
          if self.args.tgt == "zh":
            self.han1 = hanlp.load("/mnt/taurus/home/yifengliu/.hanlp/mtl/ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_xlm_base_20220608_003435", devices=1)
            self.han2 = hanlp.load("/mnt/taurus/home/yifengliu/.hanlp/tok/coarse_electra_small_20220616_012050", devices=1)
        if 'metricX' in args.model_name:
            self.min_reward = -25
            self.model_name = args.model_name
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", cache_dir="/mnt/gemini/data1/yifengliu/model")
            self.model = models.MT5ForRegression.from_pretrained(
                "google/metricx-24-hybrid-xxl-v2p6-bfloat16", torch_dtype="auto", device_map={'':0}, cache_dir="/mnt/gemini/data1/yifengliu/model"
            )
            self.max_length = args.max_len
            self.batch_size = args.batch_size
            self.training_args = transformers.TrainingArguments(
                output_dir="/mnt/gemini/data1/yifengliu/qe-lr/output/openrlhf",
                per_device_eval_batch_size=self.batch_size,
                dataloader_pin_memory=False,
            )
            self.trainer = transformers.Trainer(
                model=self.model,
                args=self.training_args,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True),
            )
        elif 'Comet' in args.model_name:
          self.min_reward = 0
          from comet import load_from_checkpoint, download_model
          self.model_name = args.model_name
          self.max_length = args.max_len
          self.batch_size = args.batch_size
          if args.model_name == "Comet22":
            model_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
          elif args.model_name == "XComet":
            model_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
          else:
            raise ValueError(f"Unsupported Comet model name: {args.model_name}")
          self.model = load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unsupported model name: {args.model_name}")

    def get_reward(self, queries, prompts, labels):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        logger.info(f"queries[0]: {queries[0]}")
        logger.info(f"queries[1]: {queries[1]}")

        scores = []
        # batch
        if "metricX" in self.model_name:
          with torch.no_grad():
              ds = []
              src_pattern = r"<\|im_start\|>user\n(.*?)Translate from (.*?) to (.*?):"
              srcs = [re.search(src_pattern, q, re.DOTALL).group(1).strip() for q in queries]

              # Match tgt between "<|im_start|>assistant\n" and "<|im_end|>"
              # tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>(.*?)<\|im_end\|>"
              if 'Qwen3' in self.base_model:
                tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>\n\n(.*?)<\|im_end\|>"
                tgts = [re.search(tgt_pattern, q, re.DOTALL).group(2).strip() for q in queries]
              else:
                tgt_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
                tgts = [re.search(tgt_pattern, q, re.DOTALL).group(1).strip() for q in queries]
              # srcs = [match.group(1).strip() for query in queries if (match := pattern.search(query))]
              # srcs = [query.split('<|im_start|>user\n', 1)[1].split(f"Translate from {lang_dict[self.src]} to {lang_dict[self.tgt]}", 1)[0].strip() for query in queries]
              # tgts = [query.split('<|im_start|>assistant\n', 1)[1].split("<|im_end|>", 1)[0].strip() for query in queries]
              # srcs = [query.split('user\n', 1)[1].split("Translate from English to Chinese", 1)[0].strip() for query in queries]
              # tgts = [query.split('Translate from English to Chinese:\nassistant\n', 1)[1] for query in queries]
              print(f"queries[0]: {queries[0]}")
              print(f"tgts[0]: {tgts[0]}")
              print(f"labels[0]: {labels[0]}")
              # print(f"src[0]: {srcs[0]}")
              # print(f"tgt[0]: {tgts[0]}")
              for src, tgt, label in zip(srcs, tgts, labels):
                  ds.append({"source": src, "hypothesis": tgt, 'reference': label})
              ds = datasets.Dataset.from_list(ds)
              if 'ref' in self.model_name:
                dataset = get_dataset(ds, self.model_name, self.tokenizer, self.max_length, self.model.device, is_qe=False)
              else:
                dataset = get_dataset(ds, self.model_name, self.tokenizer, self.max_length, self.model.device, is_qe=True)
              # import code; code.interact(local=locals())
              # print(dataset)
              print(self.model.device)
              print(dataset['input_ids'][0].device)
              predictions, _, _ = self.trainer.predict(test_dataset=dataset)
              scores.extend(-predictions)
        elif 'Comet' in self.model_name:
          ds = []
          src_pattern = r"<\|im_start\|>user\n(.*?)Translate from (.*?) to (.*?):"
          srcs = [re.search(src_pattern, q, re.DOTALL).group(1).strip() for q in queries]

          # Match tgt between "<|im_start|>assistant\n" and "<|im_end|>"
          # tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>(.*?)<\|im_end\|>"
          print(f"queries[0]: {queries[0]}")
          if 'Qwen3' in self.base_model:
            tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>\n\n(.*?)<\|im_end\|>"
            tgts = [re.search(tgt_pattern, q, re.DOTALL).group(2).strip() for q in queries]
          else:
            tgt_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
            tgts = [re.search(tgt_pattern, q, re.DOTALL).group(1).strip() for q in queries]
          # srcs = [query.split('<|im_start|>user\n', 1)[1].split(f"Translate from {lang_dict[self.src]} to {lang_dict[self.tgt]}", 1)[0].strip() for query in queries]
          # tgts = [query.split('<|im_start|>assistant\n', 1)[1].split("<|im_end|>", 1)[0].strip() for query in queries]
          inputs = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, tgts, labels)]
    
          output = self.model.predict(inputs, batch_size=8, gpus=1)
          mean_score = output.system_score
          scores.extend(output.scores)
          # scores, mean_score = output.scores, output.system_score
        # print(f"{self.model_name}: query: {queries[0]}")
        # print(f"{self.model_name}: prompt: {prompts[0]}")
        # print(f"{self.model_name}: score: {scores[0]}")
        extra_logs = {}
        extra_logs['metric_score'] = sum(scores) / len(scores)
        if self.args.rule:
          new_scores = []
          cnt = 0
          for score, tgt in zip(scores, tgts):
            # print(tgt, '\n' in tgt)
            if "\n" in tgt:
              cnt += 1
              new_scores.append(self.min_reward)
            else:
              new_scores.append(score)
          scores = new_scores
          extra_logs['rule_penalty_percent'] = cnt / len(tgts)
          
        if self.args.truncate:
          self.length_tokenizer = AutoTokenizer.from_pretrained("/mnt/gemini/data1/yifengliu/model/Qwen3-4B")
          src_length = self.length_tokenizer(srcs)['input_ids']
          tgt_length = self.length_tokenizer(tgts)['input_ids']
      
          ratio_list = [len(tgt)/len(src) for src, tgt in zip(src_length, tgt_length)]
          new_score_list = [float('inf') if (1.3 <= ratio <= 3) else self.min_reward for ratio in ratio_list]
          scores = [min(score, new_score) for score, new_score in zip(scores, new_score_list)]
        if self.args.bleu:
          bleu_score_list = []
          for tgt, label in zip(tgts, labels):
            print(f"tgt: {tgt}")
            print(f"label: {label}")
            bleu_score = get_spBLEU([tgt], [label])
            bleu_score_list.append(bleu_score)
          print(f"bleu_score_list: {bleu_score_list}")
          scores = [4*score + bleu for score, bleu in zip(scores, bleu_score_list)]
          print(f"scores: {scores}")
          extra_logs['mean_bleu_score'] = sum(bleu_score_list) / len(bleu_score_list)
          
        if self.args.align:
          print(srcs[0])
          print(tgts[0])
          if self.args.lang_detect:
            pattern = r"Translate from ([^\n<]+) to ([^\n<]+):"
            target_languages = [re.search(pattern, query).group(2).strip() for query in queries if re.search(pattern, query)]
            tgts = [tgt.replace("\n", "") for tgt in tgts]
            new_tgts = []
            for tgt, tgt_lang in zip(tgts, target_languages):
              ans = self.lang_detect_model.predict_codeswitch(tgt, beta = 20 , alpha = 3, max_lambda = 3, min_length = 10, min_prob = 0.90, max_retry=3, alpha_step_increase = 3, beta_step_increase = 5)
              ans = {key.replace("__label__", ""): value for key, value in ans.items()}
              long_lang_id = lang2long.get(tgt_lang, None)
              if long_lang_id is None:
                raise ValueError(f"Language code {long_lang_id} not found in lang2long.")
              lang_translation = ans.get(long_lang_id, None)
              # print(tgt_lang, long_lang_id, ans, lang_translation)
              if lang_translation is None:
                lang_translation = ""
              new_tgts.append(lang_translation)
            tgts = new_tgts
              
          if self.args.tgt == "zh":
            src_sentences = self.han1(srcs)['tok']
            src_sentences = [[c for c in src if c not in [",", "\"", ".", '—', "(", ")", "/", "\\", "'"]]for src in srcs]
            tgt_sentences = self.han2(tgts)
            tgt_sentences = [[c for c in tgt if c not in ["：", "。", "，", "“", "”", "（", "）", "·", "-", "/", "\\", "、"]]for tgt in tgts]
          else:
            src_sentences = [src.split() for src in srcs]
            tgt_sentences = [tgt.split() for tgt in tgts]
          align_score_list = align_score(src_sentences, tgt_sentences, self.align_model, self.align_tokenizer)
          # align_score_list = simple_align(srcs, tgts, self.aligner)
          align_score_list = [score*25 for score in align_score_list]
          print(align_score_list[:20])
          scores = [score + align_score for score, align_score in zip(scores, align_score_list)]
          extra_logs['mean_align_score'] = sum(align_score_list) / len(align_score_list)
        if self.args.lang_detect:
          pattern = r"Translate from ([^\n<]+) to ([^\n<]+):"
          target_languages = [re.search(pattern, query).group(2).strip() for query in queries if re.search(pattern, query)]
          print(tgt_pattern)
          if 'Qwen3' in self.base_model:
            tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>\n\n(.*?)<\|im_end\|>"
            tgts = [re.search(tgt_pattern, q, re.DOTALL).group(2).strip() for q in queries]
          else:
            tgt_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
            tgts = [re.search(tgt_pattern, q, re.DOTALL).group(1).strip() for q in queries]
          tgts = [tgt.replace("\n", "") for tgt in tgts]
          lang_info = self.lang_detect_model2.predict(tgts)
          min_reward = -25 if 'metricX' in self.model_name else 0
          detect_rewards = []
          cnt = 0
          for language, tgt in zip(lang_info[0], target_languages):
            lang_code = language[0].replace("__label__", "")
            pred_lang = lang_dict.get(lang_code, "")
            print(language, tgt, pred_lang, pred_lang == tgt)
            if pred_lang == tgt:
              detect_rewards.append(float('inf'))
            else:
              cnt += 1
              detect_rewards.append(min_reward)
          extra_logs['lang_penalty_percent'] = cnt / len(tgts)
          # logger.info(extra_logs[:20])
        # if self.args.lang_detect:
        #   pattern = r"Translate from ([^\n<]+) to ([^\n<]+):"
        #   target_languages = [re.search(pattern, query).group(2).strip() for query in queries if re.search(pattern, query)]
        #   tgts = [tgt.replace("\n", "") for tgt in tgts]
        #   translations = []
          
        #   for tgt, tgt_lang in zip(tgts, target_languages):
        #     ans = self.lang_detect_model.predict_codeswitch(tgt, beta = 20 , alpha = 3, max_lambda = 3, min_length = 10, min_prob = 0.90, max_retry=3, alpha_step_increase = 3, beta_step_increase = 5)
        #     ans = {key.replace("__label__", ""): value for key, value in ans.items()}
        #     long_lang_id = lang2long.get(tgt_lang, None)
        #     if long_lang_id is None:
        #       raise ValueError(f"Language code {long_lang_id} not found in lang2long.")
        #     lang_translation = ans.get(long_lang_id, None)
        #     # print(tgt_lang, long_lang_id, ans, lang_translation)
        #     if lang_translation is None:
        #       lang_translation = ""
        #     translations.append(lang_translation)
        #   original_token_length = [len(self.base_tokenizer(tgt)['input_ids']) for tgt in tgts]
        #   detect_token_length = [len(self.base_tokenizer(translation)['input_ids']) for translation in translations]
        #   ratio_list = [detect_len / orig_len for detect_len, orig_len in zip(detect_token_length, original_token_length) if orig_len > 0]
        #   detect_rewards = [self.min_reward if ratio < 0.95 else float('inf') for ratio in ratio_list]
        #   # detect_rewards = [ratio*25 for ratio in ratio_list]
        #   scores = [min(score, detect_reward) for score, detect_reward in zip(scores, detect_rewards)]
        #   # scores = [score+reward for score, reward in zip(scores, detect_rewards)]
        #   # logger.info(lang_info[0][:20])
        #   logger.info(detect_rewards[:20])
        #   extra_logs['lang_penalty_percent'] = len([reward for reward in detect_rewards if reward == self.min_reward ]) / len(tgts)
          # extra_logs['mean_lang_score'] = sum(detect_rewards) / len(detect_rewards)
          # import code; code.interact(local=locals())
        return scores, extra_logs



if __name__ == "__main__":

    parser = transformers.HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        # import code; code.interact(local=locals())
        queries = data.get("query")
        prompts = data.get("prompts")
        labels = data.get("labels", None)
        # import code; code.interact(local=locals())
        rewards, extra_logs = reward_model.get_reward(queries, prompts, labels)
        # rewards = torch.tensor([float(reward) for reward in rewards])
        rewards = [float(reward) for reward in rewards]
        result = {"rewards": rewards, "scores": rewards, "extra_logs": extra_logs}
        logger.info(f"Sent JSON: {result['rewards'][:20]}")
        return JSONResponse(result)
    print(args.rule, args.lang_detect)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
