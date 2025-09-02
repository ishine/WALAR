# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluates the predictions from a MetricX model."""

import collections
import dataclasses
import json
import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import code.models as models
import models

from typing import Optional, Tuple, Union, List
import scipy
from scipy import stats
from mt_metrics_eval import data
from mt_metrics_eval import meta_info
from mt_metrics_eval import tasks
import torch
import numpy as np
import itertools
import tqdm
from tqdm import *
from vllm import LLM, SamplingParams
import transformers
import datasets

from transformers import AutoTokenizer, AutoModel
from typing import Optional, List, Dict
from utils import preprocess_dataset, mm_dict, lang_dict, RewardModel
from mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer, TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING, validate_number
from mqm_utils import TEMPLATE_DA, extract_boxed_number


@dataclasses.dataclass
class Arguments:
  wmt_year: int = dataclasses.field(
      default=24,
      metadata={"help": "The WMT year to evaluate."},
  )
  
  dtype: str = dataclasses.field(
      default="fp32",
      metadata={
          "help": "The data type to use for the model. "
                  "Supported types: 'fp16', 'bf16'."
      },
  )
  
  model_name: str = dataclasses.field(
      default="metricx",
      metadata={
          "help": "The name or path of the model to use for evaluation. "
                  "Supported models: 'metricx', 'XComet'."
      },
  )
  
  output_dir: str = dataclasses.field(
      metadata={"help": "The output directory with evaluation metrics."},
      default="/dev/null",
  )
  
  model_size: str = dataclasses.field(
      default="xl",
      metadata={
          "help": "The size of the model to use for evaluation. "
                  "Supported sizes: 'xxl', 'xl'."
      },
  )
  
  turns: int = dataclasses.field(
      default=1,
      metadata={
          "help": "The number of turns to use for the evaluation. "
                  "This is only used for Qwen models."
      },
  )
  
  eval_type: str = dataclasses.field(
      default="mqm",
      metadata={
          "help": "The type of evaluation to perform. "
                  "Supported types: 'mqm', 'esa', 'da'."
      },
  )
  
  alignment: bool = dataclasses.field(
      default=False,
      metadata={
          "help": "Whether to use alignment information. "
                  "This is only used for ESA evaluation."
      },
  )
  
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
    if model_name == 'metricX':
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
      src = example.pop("source", None)
      mt = example.pop("hypothesis", None)
      if src is None or mt is None:
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
        padding=False,
    )

  def _remove_eos(example):
    example["input_ids"] = example["input_ids"][:-1]
    example["attention_mask"] = example["attention_mask"][:-1]
    return example
  
  if model_name == "metricX":
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

def load_tokenizer_and_model(metric_name: str, model_size: str, model_dtype:str, device: torch.device) -> Tuple[transformers.PreTrainedTokenizer, transformers.PreTrainedModel]:
  """Loads the tokenizer and model for the specified metric."""
  tokenizer, model = None, None
  if metric_name == 'metricX':
    path_dict = {
      "xl": {
        "fp32": "google/metricx-24-hybrid-xl-v2p6",
        "bf16": "google/metricx-24-hybrid-xl-v2p6-bfloat16",
      },
      "xxl": {
        "fp32": "google/metricx-24-hybrid-xxl-v2p6",
        "bf16": "google/metricx-24-hybrid-xxl-v2p6-bfloat16",
      }
    }
    path = path_dict[model_size][model_dtype]
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", cache_dir="/mnt/gemini/data1/yifengliu/model")
    model = models.MT5ForRegression.from_pretrained(
       path, torch_dtype="auto", device_map="auto", cache_dir="/mnt/gemini/data1/yifengliu/model"
    )
    model.eval()
  elif 'Comet' in metric_name:
    path_dict = {
      "XComet": "/mnt/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt",
      "Comet-qe-da": "/mnt/data1/yifengliu/model/models--Unbabel--wmt20-comet-qe-da/snapshots/2e7ffc84fb67d99cf92506611766463bb9230cfb/checkpoints/model.ckpt",
    }
    model_path = path_dict.get(metric_name, None)
    from comet import download_model, load_from_checkpoint
    print(f"Loading model from {model_path}")
    model = load_from_checkpoint(model_path)
    model.eval()
  elif 'Qwen' in metric_name:
    path_dict = {
      "Qwen3-32B": "/mnt/gemini/data1/yifengliu/model/Qwen3-32B",
      "Qwen3-32B-AWQ": "/mnt/gemini/data1/yifengliu/model/Qwen3-32B-AWQ",
      "Qwen3-235B": "/mnt/gemini/data1/yifengliu/model/Qwen3-235B-A22B-GPTQ-Int4",
      "Qwen3-30B-A3B": "/mnt/gemini/data1/yifengliu/model/Qwen3-30B-A3B-Instruct-2507",
    }
    if metric_name == 'Qwen3-235B':
      tensor_parallel_size = 4
    elif metric_name == 'Qwen3-32B-AWQ':
      tensor_parallel_size = 1
    elif metric_name == 'Qwen3-32B':
      tensor_parallel_size = 4
    else:
      raise ValueError(f"Unsupported metric name: {metric_name}")
    model_path = path_dict.get(metric_name, None)
    model = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, task="generate", enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  elif 'Seed' in metric_name:
    path_dict = {
      "Seed-X-RM-7B": "/mnt/gemini/data1/yifengliu/model/Seed-X-RM-7B",
    }
    model_path = path_dict.get(metric_name, None)
    model = RewardModel(model_path)
    tokenizer = None
  else:
    raise ValueError(f"Unsupported metric name: {metric_name}")
  # model.to(device)
  # model.parallelize() 
  return tokenizer, model

def get_sampling_params(metric_name, args):
  if metric_name == "Qwen3-235B":
      sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5, max_tokens=2048, n=args.turns)
  elif metric_name == "Qwen3-32B-AWQ":
      sampling_params = SamplingParams(temperature=1, top_p=0.9, top_k=-1, min_p=0, max_tokens=2048, n=args.turns)
  else:
      # sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=args.max_tokens, n=args.turns)
      sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=-1, presence_penalty=0, frequency_penalty=0, max_tokens=2048, n=args.turns)
  return sampling_params

def get_scores(eval_type: str, ds: List[Dict], sampling_params: SamplingParams, model: LLM, tokenizer: AutoTokenizer=None, src_lang="English", tgt_lang="Chinese"):
    # source_lang, source_seg, target_lang, target_seg
    prompts = []
    for data in ds:
        temp_data = {
            'source_lang': src_lang,
            'target_lang': tgt_lang,
            'source_seg': data['source'],
            'target_seg': data['hypothesis'],
        }        
        if eval_type == "mqm":
            prompt = apply_template(TEMPLATE_GEMBA_MQM, temp_data)
        elif eval_type == "esa":
            prompt = apply_template(TEMPLATE_GEMBA_ESA_ERROR_SPANS, temp_data)
        elif eval_type == "da":
            prompt = apply_template(TEMPLATE_DA, temp_data)
            prompt = [{"role": "user", "content": prompt}]
        
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(prompt)
            
    outputs = model.generate(prompts, sampling_params=sampling_params)
    # outputs = model.chat(
    #     prompts, 
    #     sampling_params,
    #     chat_template_kwargs={"enable_thinking": False},  # Set to False to strictly disable thinking
    # )
    outputs1 = [[opt.text for opt in output.outputs] for output in outputs]
    if eval_type == "mqm":
        scores = [[parse_mqm_answer(opt) for opt in output] for output in outputs1]
    elif eval_type == "esa":
        print("Evaluating ESA Error Spans...")
        prompts = []
        n1 = len(outputs1)
        outputs1 = [opt for output in outputs1 for opt in output]
        turns = len(outputs1) // n1
        for data, error_span in zip(ds, outputs1):
            temp_data = {
                'source_lang':  src_lang,
                'target_lang':  tgt_lang,
                'source_seg':   data['source'],
                'target_seg':   data['hypothesis'],
                "error_spans":  error_span,
            }
            prompt = apply_template(TEMPLATE_GEMBA_ESA_RANKING, temp_data)
            prompt = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(prompt)
        outputs = model.generate(prompts, sampling_params=sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        # scores  = [validate_number(output) for output in outputs]
        scores = [extract_boxed_number(output) for output in outputs]
        scores = [scores[i:i+turns] for i in range(0, len(scores), turns)]
    elif eval_type == "da":
        # scores = [extract_boxed_number(output) for output in outputs1]
        scores = [[extract_boxed_number(opt) for opt in output] for output in outputs1]
    
    # scores = [score if score is not None else 0 for score in scores]
    scores = [[0 if sc is None else sc for sc in s]for s in scores]
    scores = [sum(s) / len(s) for s in scores]
    # import code; code.interact(local=locals())
    return scores

def get_langs(src, tgt):
    src_lang, tgt_lang = mm_dict.get(src, ''), mm_dict.get(tgt, '')
    if len(src_lang) == 0 or len(tgt_lang) == 0:
        src_lang, tgt_lang = lang_dict.get(src, ''), lang_dict.get(tgt, '')
    # The case for IndicMT
    if tgt_lang == '':
        tgt_lang = tgt.capitalize()
        # raise ValueError(f"Unsupported language codes: {src}, {tgt}")
    print(f"Source language: {src_lang}, Target language: {tgt_lang}")
    return src_lang, tgt_lang

def get_predictions(
    metric_name: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    srcs: List[str],
    hyps: List[str],
    refs: List[str],
    lp: str,
    args: Arguments,
) -> np.ndarray:
  predictions = None
  if metric_name == 'metricX':
    ds = [{"source": src, "hypothesis": hyp, "reference": ref} for src, hyp, ref in zip(srcs, hyps, refs)]
    ds = datasets.Dataset.from_list(ds)
    ds = get_dataset(ds, metric_name, tokenizer, max_input_length=1536, device='cuda', is_qe=True)
    training_args = transformers.TrainingArguments(
      # output_dir="/dev/null",
      per_device_eval_batch_size=1,
      dataloader_pin_memory=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
    )
    predictions, a, b = trainer.predict(test_dataset=ds)
  elif 'Comet' in metric_name:
    ds = [{"source": src, "hypothesis": hyp, "reference": ref} for src, hyp, ref in zip(srcs, hyps, refs)]
    ds = datasets.Dataset.from_list(ds)
    ds = get_dataset(ds, metric_name, tokenizer, max_input_length=1536, device='cuda:0', is_qe=True)
    ds = list(ds)
    inputs = [data['input'] for data in ds]
    model_output = model.predict(inputs, batch_size=1, gpus=torch.cuda.device_count())
    predictions = model_output.scores
    predictions = np.array(predictions)
  elif 'Qwen' in metric_name:
    src, tgt = lp.split('-')[0], lp.split('-')[1]
    ds = [{"source": src, "hypothesis": hyp} for src, hyp in zip(srcs, hyps)]
    sampling_params = get_sampling_params(metric_name, args)
    src_lang, tgt_lang = get_langs(src, tgt)
    predictions = get_scores(args.eval_type, ds, sampling_params, model, tokenizer, src_lang, tgt_lang)
  elif 'Seed' in metric_name:
    src, tgt = lp.split('-')[0], lp.split('-')[1]
    src_lang, tgt_lang = get_langs(src, tgt)
    prompt = [f"Translate the following {src_lang} sentence into {tgt_lang}:\n{src} <{tgt}>" for src in srcs]
    candidate = hyps
    print(f"prompt: {prompt[:2]}")
    print(f"candidate: {candidate[:2]}")
    predictions = model.score(prompt, candidate, 8) 
    # ds = datasets.Dataset.from_list(ds)
  return predictions

def NewMetric(
    metric_name: str,
    model_size: str,
    model_dtype: str,
    lp: str,
    domains: dict[str, list[list[int]]],
    docs: dict[str, list[int]],
    src: list[str],
    ref: list[str],
    hyps: dict[list[str]],
    args: Arguments,
) -> dict[str, list[float]]:
  """
  Generate metric scores.

  Args:
    level: Level for which to produce scores, 'sys' or 'seg'.
    lp: Language pair, eg 'en-de'.
    domains: Map from domain name to [[beg, end+1], ...] segment position lists.
    docs: Map from doc name to [beg, end+1] segment positions.
    src: List of source segments.
    ref: List of reference segments.
    hyps: Map from MT system name to output segments for that system.

  Returns:
    Map from system name to scores, a list of segment-level scores if level is
    'seg', or a list containing a single score if level is 'sys'.
  """
  # Sample metric just computes a length match between each hypothesis and the
  # reference. It ignores lp, domains, docs, and source.

  del domains, docs

  seg_scores = {}
  sys_scores = {}
  tokenizer, model = load_tokenizer_and_model(metric_name, model_size, model_dtype, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
  if args.alignment:
    align_path = "/mnt/gemini/data1/yifengliu/model/bge-m3"
    align_model = AutoModel.from_pretrained(align_path)
    align_tokenizer = AutoTokenizer.from_pretrained(align_path)
    align_model.to("cuda:0")
  for sysname, hyp in hyps.items():
    print(f"Evaluating {metric_name}-{model_size}-{model_dtype} for system {sysname} on {lp}...")
    # src, hyp, ref = src[:len(src)//100], hyp[:len(hyp)//100], ref[:len(ref)//100]
    predictions = get_predictions(metric_name, model, tokenizer, src, hyp, ref, lp, args)
    if args.alignment:
      srcs, tgts = src, hyp
      srcs = [src.split() for src in srcs]
      tgts = [tgt.split() for tgt in tgts]
      scores = align_score(srcs, tgts, align_model, align_tokenizer, batch_size=16)
      predictions = np.array([prediction + score for prediction, score in zip(predictions, scores)])
    if metric_name == 'metricX':
      seg_scores[sysname] = -predictions
      sys_scores[sysname] = [-predictions.mean()]
    elif 'Comet' in metric_name:
      seg_scores[sysname] = predictions
      sys_scores[sysname] = [predictions.mean()]
    else:
      seg_scores[sysname] = predictions
      sys_scores[sysname] = [np.mean(predictions)]
  return seg_scores, sys_scores

def get_lps(wmt_year: int) -> list[str]:
  """Returns the list of language pairs for the given WMT year."""
  if wmt_year == 24:
    return ["en-de", "en-es", "ja-zh", "cs-uk", "en-cs", "en-hi", "en-is","en-ja", "en-ru", "en-uk", "en-zh"]
  elif wmt_year == 23:
    return ["cs-uk", "en-cs", "en-he", "en-ru", "en-zh", "ja-en", "uk-en", "de-en", "en-de", "en-ja", "en-uk", "he-en", "ru-en", "zh-en"]
  else:
    raise ValueError(f"Unsupported WMT year: {wmt_year}")

def get_meta_info(wmt_year: int) -> meta_info.MetaInfo:
  """Returns the meta info for the given WMT year."""
  if wmt_year == 24:
    return meta_info.WMT24
  elif wmt_year == 23:
    return meta_info.WMT23
  else:
    raise ValueError(f"Unsupported WMT year: {wmt_year}")

def get_tasks(wmt_year: int, lps: list[str], k: int = 0) -> Tuple[tasks.Task, dict]:
  """Returns the tasks and weights for the given WMT year and language pairs."""
  if wmt_year == 24:
    return tasks.WMT24(lps, k)
  elif wmt_year == 23:
    return tasks.WMT23(lps, k)
  else:
    raise ValueError(f"Unsupported WMT year: {wmt_year}")

def write_result(metric_name: str, model_size: str, lp: str, seg_scores: dict[str, list[float]], sys_scores: dict[str, list[float]], output_dir: str, args: Arguments) -> None:
  if 'Qwen' not in metric_name:
    output_dir = os.path.join(output_dir, metric_name + '-' + model_size)
  else:
    output_dir = os.path.join(output_dir, metric_name + "-" + args.eval_type + "-" + str(args.turns))
  seg_file = os.path.join(output_dir, "seg", f"{lp}.jsonl")
  sys_file = os.path.join(output_dir, "sys", f"{lp}.jsonl")
  os.makedirs(os.path.dirname(seg_file), exist_ok=True)
  os.makedirs(os.path.dirname(sys_file), exist_ok=True)
  with open(seg_file, "w") as f:
    for sysname, scores in seg_scores.items():
      for score in scores:
        score = np.array(score, dtype=np.float16)
        score = score.astype(float).tolist()
        f.write(json.dumps({"system": sysname, "score": score}) + "\n")
  with open(sys_file, "w") as f:
    for sysname, scores in sys_scores.items():
      for score in scores:
        score = np.array(score, dtype=np.float16)
        score = score.astype(float).tolist()
        f.write(json.dumps({"system": sysname, "score": score}) + "\n")

def evaluate(lps: list[str], model_scores: dict[list[int]], sy_scores: dict[list[int]]):
  for lp in lps:
    model_score, sys_score = model_scores[lp], sy_scores[lp]
    mask = ~np.isnan(sys_score)
    pearsonr, _ = stats.pearsonr(
      model_score[mask],
      sys_score[mask],
    )
    kendalltau, _ = stats.kendalltau(
      model_score[mask],
      sys_score[mask],
    )
    spearmanr, _ = stats.spearmanr(
      model_score[mask],
      sys_score[mask],
    )
    print(f"Language pair: {lp}")
    print(f"Pearson correlation: {pearsonr:.4f}")
    print(f"Kendall tau correlation: {kendalltau:.4f}")
    print(f"Spearman correlation: {spearmanr:.4f}")
    print("=========================================")

def main() -> None:
  parser = transformers.HfArgumentParser(Arguments)
  (args,) = parser.parse_args_into_dataclasses()

  lps = get_lps(args.wmt_year)
  baseline_metainfo = get_meta_info(args.wmt_year)
  evs_dict = {(f'wmt{args.wmt_year}', lp): data.EvalSet(f'wmt{args.wmt_year}', lp, True, path="/mnt/gemini/home/yifengliu/.mt-metrics-eval/mt-metrics-eval-v2") for lp in lps}
  model_scores, sy_scores = {}, {}
  # ESA
  # lps = ["en-cs", "en-hi", "en-is"]
  lps = ["cs-uk"]
  
  # MQM
  # lps = ["en-de", "en-es", "ja-zh"]
  # lps = ["en-es"]
  # lps = ["ja-zh"]
  # lps = ["en-de", "en-es"]
  
  for lp in lps:
    evs = evs_dict[(f'wmt{args.wmt_year}', lp)]
    # gold_scores = evs.Scores("seg", "mqm")
    gold_scores = evs.Scores("seg", "esa")
    # import code; code.interact(local=locals())
    for refname, ref in evs.all_refs.items():
      seg_scores, sys_scores = NewMetric(
          args.model_name, args.model_size, args.dtype, evs.lp, evs.domains, evs.docs, evs.src, ref, evs.sys_outputs, args)
      evs.AddMetric(args.model_name, {refname}, 'sys', sys_scores, replace=True)
      evs.AddMetric(args.model_name, {refname}, 'seg', seg_scores, replace=True)
    # import code; code.interact(local=locals())
    m_scores, s_scores = [], []
    keys = sorted(seg_scores.keys())
    m_value, s_value = [seg_scores[key] for key in keys], [gold_scores[key] for key in keys]
    for m, s in zip(m_value, s_value):
      if len(m) == 0 and len(s) == 0:
        continue
      m_scores.extend(m)
      s_scores.extend(s)
    model_scores[lp] = np.array(m_scores, dtype=np.float32)
    sy_scores[lp] = np.array(s_scores, dtype=np.float32)
    
  # import code; code.interact(local=locals())
  # Add new metric to the primary lists, so it will get picked up when tasks get
  # run with primary=True (avoiding having to evaluate all contrastive
  # submissions as well).
  
  for lp in lps:
    write_result(args.model_name, args.model_size, lp, seg_scores, sys_scores, args.output_dir, args)
  # import code; code.interact(local=locals())
  for evs in evs_dict.values():
    evs.SetPrimaryMetrics(evs.primary_metrics | {args.model_name})


  # Set k=1000 for a more realistic comparison
  tasks, wts = get_tasks(args.wmt_year, lps, k=0)

  # Takes about 3 minutes.
  new_results = tasks.Run(eval_set_dict=evs_dict)
  
  avg_corrs = new_results.AverageCorrs(wts)

  table = new_results.Table(
      metrics=list(avg_corrs),
      initial_column=avg_corrs,
      initial_column_header='avg-corr',
      attr_list=['lang', 'level', 'corr_fcn'],
      nicknames={'KendallWithTiesOpt': 'acc-t'},
      fmt='text',
      baselines_metainfo=baseline_metainfo)

  print(table)
  # import code; code.interact(local=locals())
  evaluate(lps, model_scores, sy_scores)
  
  
  
  import code; code.interact(local=locals())

if __name__ == "__main__":
  main()