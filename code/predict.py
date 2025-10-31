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
"""Runs inference with a MetricX model."""

import dataclasses
import json
import pandas as pd
import csv
import os
import sys
import tqdm
import itertools
from tqdm import *

sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
from utils import write_to_file, preprocess_dataset, my_load_dataset, three2two, two2three
from utils import mm_dict, lang_dict
import models
import datasets
from typing import Optional, Tuple, Union, List
# from code import models
import torch
import transformers
import fasttext
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModel


@dataclasses.dataclass
class Arguments:
  """Prediction command-line arguments."""
  model_name: str = dataclasses.field(
      metadata={
          "help": "The name or path of the model to use for prediction. "
                  "Supported models: 'metricX', 'XComet'."
      }
  )
  
  max_input_length: int = dataclasses.field(
      metadata={"help": "The maximum allowable input sequence length."},
  )

  batch_size: int = dataclasses.field(
      metadata={"help": "The global prediction batch size."},
  )

  input_file: str = dataclasses.field(metadata={"help": "The input file."})

  output_dir: str = dataclasses.field(
      metadata={"help": "The output directory with predictions."},
  )
  
  tgt: str = dataclasses.field(
      metadata={"help": "The target language."},
      default="zh",
  )
  
  model_size: str = dataclasses.field(
      metadata={
          "help": "The size of the model to use for prediction. "
                  "Supported sizes: 'xxl', 'xl'."
      },
      default="xl"
  )
  
  dtype: str = dataclasses.field(
      metadata={"help": "The data type of the model."},
      default="fp32",
  )
  
  src: str = dataclasses.field(
      metadata={"help": "The source language."},
      default="en",
  )
  
  src_list: Optional[List[str]] = dataclasses.field(
      metadata={"help": "List of source languages for batch processing."},
      default=None,
  )
  
  tgt_list: Optional[List[str]] = dataclasses.field(
      metadata={"help": "List of target languages for batch processing."},
      default=None,
  )

  qe: bool = dataclasses.field(
      metadata={"help": "Indicates the metric is a QE metric."},
      default=False,
  )
  
  alignment: bool = dataclasses.field(
      metadata={"help": "Indicates whether to output word-level alignment."},
      default=False,
  )
  
  lang: bool = dataclasses.field(
      metadata={"help": "Indicates whether to do language filtering."},
      default=False,
  )
  
def get_tokenizer_and_model(
    model_name: str,
    model_size: str,
    model_dtype: str
):
  tokenizer, model = None, None
  if model_name == "metricX":
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
    # path = path_dict.get(model_size, None)
    path = path_dict[model_size][model_dtype]
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", cache_dir="/mnt/gemini/data1/yifengliu/model")
    model = models.MT5ForRegression.from_pretrained(
        path, torch_dtype="auto", device_map="auto", cache_dir="/mnt/gemini/data1/yifengliu/model"
    )
  elif "Comet" in model_name:
    path_dict = {
      "XComet": {
        "xl": "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt",
        "xxl": "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XXL/snapshots/873bac1b1c461e410c4a6e379f6790d3d1c7c214/checkpoints/model.ckpt"
      },
      "Comet-qe-da": "/mnt/gemini/data1/yifengliu/model/models--Unbabel--wmt20-comet-qe-da/snapshots/2e7ffc84fb67d99cf92506611766463bb9230cfb/checkpoints/model.ckpt",
      "Cometkiwi": "/mnt/gemini/data1/yifengliu/model/wmt22-cometkiwi-da/checkpoints/model.ckpt"
    }
    from comet import download_model, load_from_checkpoint
    # model_path = download_model("Unbabel/XCOMET-XXL", "/mnt/data1/yifengliu/model")
    model_path = path_dict[model_name][model_size] if isinstance(path_dict[model_name], dict) else path_dict[model_name]
    # model_path = path_dict.get(model_name, None)
    print(f"Loading model from {model_path}")
    model = load_from_checkpoint(model_path)
    # model = torch.load(model_path, weights_only=False)
  if tokenizer is None and model is None:
    raise ValueError("Unsupported model name or path: {}".format(model_name))
  return tokenizer, model

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
      src = example.pop("source", "")
      mt = example.pop("hypothesis", "")
      # if src == "" or mt == "":
      #   raise ValueError(
      #       "Input data must have 'source' and 'hypothesis' fields for Comet models."
      #   )
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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
  elif "Comet" in model_name:
    ds = ds.map(_make_input)
    data_collator = None

  return ds, data_collator

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

def get_predictions(
    ds: datasets.Dataset,
    model: Union[transformers.PreTrainedModel, models.MT5ForRegression],
    data_collator: Optional[DataCollatorWithPadding],
    per_device_batch_size: int,
    model_name: str,
    output_dir: Optional[str] = None,
) -> Union[List[float], pd.DataFrame]:
  """Gets the predictions for the dataset.

  Args:
    ds: The dataset.
    model: The model to use for prediction.
    per_device_batch_size: The batch size to use for prediction.
    model_name: The name of the model.
    output_dir: The output directory.

  Returns:
    The predictions.
  """
  if model_name == "metricX":
    training_args = transformers.TrainingArguments(
      output_dir=output_dir,
      per_device_eval_batch_size=per_device_batch_size,
      dataloader_pin_memory=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )
    predictions, _, _ = trainer.predict(test_dataset=ds)
  elif "Comet" in model_name:
    # predictions = model.predict(ds, batch_size=per_device_batch_size).predictions
    ds = list(ds)
    inputs = [data['input'] for data in ds]
    # import code; code.interact(local=locals())
    model_output = model.predict(inputs, batch_size=per_device_batch_size, gpus=torch.cuda.device_count())
    predictions = model_output.scores
  else:
    raise ValueError("Unsupported model name or path: {}".format(model_name))
  # import code; code.interact(local=locals())
    
  return predictions

def load_flores(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  return lines

def load_benchmax_json(file_path, src_lang, tgt_lang):
  """Load JSON file for benchmax data and return dataset.
  
  Args:
    file_path: Path to the benchmax JSON file (contains only outputs)
    src_lang: Source language code (e.g., 'en')
    tgt_lang: Target language code (e.g., 'cs')
  
  Returns:
    List of dictionaries with 'source', 'hypothesis', and 'reference' fields
  """
  # Load the benchmax JSON file (contains only outputs)
  with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  
  # Get the translated outputs
  outputs = data.get('outputs', [])
  try:
    output = outputs[0].get("score")
    outputs = [item["text"] for item in outputs]
  except:
    pass
  # if len(outputs) > 0 and outputs[0].get("score") is not None:
    # outputs = [item["text"] for item in outputs]
  # Load source sentences from flores dataset
  flores_dir = "/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"
  src_file = os.path.join(flores_dir, f"{two2three[src_lang]}.devtest")
  tgt_file = os.path.join(flores_dir, f"{two2three[tgt_lang]}.devtest")
  
  if not os.path.exists(src_file):
    raise FileNotFoundError(f"Source file not found: {src_file}")
  if not os.path.exists(tgt_file):
    raise FileNotFoundError(f"Target file not found: {tgt_file}")
  
  with open(src_file, 'r', encoding='utf-8') as f:
    source_sentences = [line.strip() for line in f.readlines()]
  
  with open(tgt_file, 'r', encoding='utf-8') as f:
    reference_sentences = [line.strip() for line in f.readlines()]
  
  # Ensure we have the same number of sentences
  if len(source_sentences) != len(outputs):
    raise ValueError(f"Mismatch in sentence counts: {len(source_sentences)} source vs {len(outputs)} target")
  if len(reference_sentences) != len(outputs):
    raise ValueError(f"Mismatch in sentence counts: {len(reference_sentences)} reference vs {len(outputs)} target")
  
  # Create dataset
  ds = []
  for src, ref, hyp in zip(source_sentences, reference_sentences, outputs):
    ds.append({
      "source": src,
      "reference": ref,
      "hypothesis": hyp
    })
  
  return ds

def save_benchmax_results(file_path, ds, predictions, model_name):
  """Save benchmax results back to the same JSON file with scores added."""
  # Load the original JSON file
  with open(file_path, 'r', encoding='utf-8') as f:
    original_data = json.load(f)
  
  
  # Add overall score information
  mean_score = sum(predictions) / len(predictions)
  if model_name == "metricX":
    original_data["metricx_score"] = float(mean_score)
  elif model_name == "XComet":
    original_data["xcomet_score"] = float(mean_score)
  else:
    original_data[f"{model_name.lower()}_score"] = float(mean_score)
  
  # Save back to the same file
  with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(original_data, f, ensure_ascii=False, indent=2)
  
  print(f"Saved {len(predictions)} scores to {file_path}")
  print(f"{model_name} Score: {mean_score:.4f}")

def process_language_pairs(
    src_list: List[str], 
    tgt_list: List[str], 
    args: Arguments,
    tokenizer,
    model,
    device,
    per_device_batch_size
) -> None:
  """Process multiple language pairs in batch.
  
  Args:
    src_list: List of source languages
    tgt_list: List of target languages  
    args: Arguments object with model configuration
    tokenizer: Model tokenizer
    model: Model for prediction
    device: Device to run on
    per_device_batch_size: Batch size per device
  """
  # Generate all possible language pair combinations
  language_pairs = []
  for src in src_list:
    for tgt in tgt_list:
      if src != tgt:  # Skip same language pairs
        language_pairs.append((src, tgt))
  
  print(f"Processing {len(language_pairs)} language pairs:")
  for src, tgt in language_pairs:
    print(f"  {src} -> {tgt}")
  
  # Process each language pair
  for src, tgt in language_pairs:
    print(f"\nProcessing language pair: {src} -> {tgt}")
    
    # Generate input file path based on the pattern
    input_file = generate_input_file_path(args.input_file, src, tgt)
    print(f"Input file: {input_file}")
    # Update args for this language pair
    current_args = dataclasses.replace(args, src=src, tgt=tgt, input_file=input_file)
    
    # Process the language pair
    process_single_language_pair(current_args, tokenizer, model, device, per_device_batch_size)

def generate_input_file_path(input_file_pattern: str, src: str, tgt: str) -> str:
  """Generate input file path based on pattern and language pair.
  
  Args:
    input_file_pattern: Base pattern for input files
    src: Source language
    tgt: Target language
    
  Returns:
    Complete input file path
  """
  # Determine file extension and format based on pattern
  if "afriMTE" in input_file_pattern:
    return f"{input_file_pattern}.{src}-{tgt}.jsonl"
  elif "IndicMT" in input_file_pattern:
    return f"{input_file_pattern}/{tgt}.jsonl"
  elif "wmt23-dev" in input_file_pattern:
    return f"{input_file_pattern}.{src}{tgt}.df.short.tsv"
  elif "wmt24-test" in input_file_pattern:
    return f"{input_file_pattern}/{src}-{tgt}.jsonl"
  elif "low-res" in input_file_pattern:
    return f"{input_file_pattern}/{src}-{tgt}.csv"
  elif "flores101_dataset/devtest" in input_file_pattern:
    # For flores_devtest, we use the base directory directly
    # The actual file paths are constructed in process_single_language_pair
    return input_file_pattern
  elif "BenchMAX" in input_file_pattern:
    return f"{input_file_pattern}/result_{src}-{tgt}.json"
  elif "flores" in input_file_pattern:
    return f"{input_file_pattern}/{src}-{tgt}.txt"
  else:
    # Default pattern
    return f"{input_file_pattern}/{src}-{tgt}.jsonl"

def has_content(model, file_path):
  target_content = None
  if model == "metricX":
    target_content = "MetricX Score: "    
  elif model == "XComet":
    target_content = "XComet Score: "
  else:
    raise ValueError("Unsupported model name or path: {}".format(model))
  with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
      if target_content in line:
        return True
  return False

def has_content2(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def process_single_language_pair(
    args: Arguments,
    tokenizer,
    model,
    device,
    per_device_batch_size
) -> None:
  """Process a single language pair.
  
  Args:
    args: Arguments object with model configuration
    tokenizer: Model tokenizer
    model: Model for prediction
    device: Device to run on
    per_device_batch_size: Batch size per device
  """
  model.eval()
  # Handle flores_devtest case specially
  if 'flores101_dataset/devtest' in args.input_file:
    # For flores_devtest, we need to read two separate files and combine them
    base_dir = args.input_file
    src_file = os.path.join(base_dir, f"{args.src}.devtest")
    tgt_file = os.path.join(base_dir, f"{args.tgt}.devtest")
    
    print(f"Loading source file: {src_file}")
    print(f"Loading target file: {tgt_file}")
    
    # Check if files exist
    if not os.path.exists(src_file):
      raise FileNotFoundError(f"Source file not found: {src_file}")
    if not os.path.exists(tgt_file):
      raise FileNotFoundError(f"Target file not found: {tgt_file}")
    
    src_dataset = load_flores(src_file)
    tgt_dataset = load_flores(tgt_file)
    
    if len(src_dataset) != len(tgt_dataset):
      raise ValueError(f"Source and target datasets have different lengths: {len(src_dataset)} vs {len(tgt_dataset)}")
    
    print(f"Successfully loaded {len(src_dataset)} sentence pairs for {args.src} -> {args.tgt}")
    
    ds = [
      {
        "source": src_dataset[i].strip(),
        "hypothesis": tgt_dataset[i].strip(),
      } for i in range(len(src_dataset))
    ]
    name = "flores_devtest"
  elif 'BenchMAX' in args.input_file:
    # Handle benchmax JSON files
    print(f"Loading benchmax JSON file: {args.input_file}")
    
    # Check if file exists
    if not os.path.exists(args.input_file):
      raise FileNotFoundError(f"Benchmax file not found: {args.input_file}")
    
    ds = load_benchmax_json(args.input_file, args.src, args.tgt)
    print(f"Successfully loaded {len(ds)} sentence pairs for {args.src} -> {args.tgt}")
    name = "benchmax"
  else:
    ds, name = preprocess_dataset(args.input_file)
  dirname = args.output_dir
  if not args.alignment:
    dirname = os.path.join(dirname, args.model_name + "-" + args.model_size + "-" + args.dtype)
  else:
    dirname = os.path.join(dirname, args.model_name + "-" + args.model_size + "-" + args.dtype + "-align")
  if name != "flores":
    if name == "benchmax":
      with open(args.input_file, 'r') as f:
        data = json.load(f)
      if data.get('xcomet_score', None) is not None and args.model_name == "XComet":
        print(f"Benchmax file {args.input_file} already has XComet score. Skipping...")
        return
        
    else:
      if dirname:
        os.makedirs(dirname, exist_ok=True)
      output_file = os.path.join(
          dirname,
          f"{args.src}-{args.tgt}.jsonl",
      )
      if has_content2(output_file):
        print(f"Output file {output_file} already exists and is non-empty. Skipping...")
        return
  dt = datasets.Dataset.from_list(ds)
  dt, data_collator = get_dataset(
      dt,
      args.model_name,
      tokenizer,
      args.max_input_length,
      device,
      args.qe,
  )
  predictions = get_predictions(dt, model, data_collator, per_device_batch_size, args.model_name, args.output_dir)
  
  if args.alignment:
    align_path = "/mnt/gemini/data1/yifengliu/model/bge-m3"
    align_model = AutoModel.from_pretrained(align_path)
    align_tokenizer = AutoTokenizer.from_pretrained(align_path)
    align_model.to("cuda:0")
    srcs, tgts = [d['source'] for d in ds], [d['hypothesis'] for d in ds]
    src_copy = [src.split() for src in srcs]
    tgt_copy = [tgt.split() for tgt in tgts]
    scores = align_score(src_copy, tgt_copy, align_model, align_tokenizer, batch_size=16)
    # import code; code.interact(local=locals())
    predictions = [prediction + score*25 for prediction, score in zip(predictions, scores)]
  
  if args.lang:
    srcs, tgts = [d['source'] for d in ds], [d['hypothesis'] for d in ds]
    lang_detect_model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
    lang_info = lang_detect_model.predict(tgts)
    lang_id = [language[0].replace("__label__", "") for language in lang_info[0]]
    new_score_list = [-25 if lang != args.tgt else float("inf") for lang in lang_id]
    predictions = [min(prediction, score) for prediction, score in zip(predictions, new_score_list)]
  
  # Save results
  print(f"prediction: {sum(predictions)/len(predictions)}")
  # import code; code.interact(local=locals())
  if name == "benchmax":
    # For benchmax, save results back to the same JSON file
    save_benchmax_results(args.input_file, ds, predictions, args.model_name)
  elif name != "flores":
    if dirname:
      os.makedirs(dirname, exist_ok=True)
    output_file = os.path.join(
        dirname,
        f"{args.src}-{args.tgt}.jsonl",
    )
    write_to_file(output_file, ds, predictions, args.model_name)
  else:
    with open(args.input_file, 'a') as f:
      mean_score = sum(predictions) / len(predictions)
      if args.model_name == "metricX":
        f.write(f"MetricX Score: {mean_score:.4f}\n")
        print(f"{args.src}-{args.tgt}: MetricX Score: {mean_score:.4f}")
      if args.model_name == "XComet":
        f.write(f"XComet Score: {mean_score:.4f}\n")
        print(f"{args.src}-{args.tgt}: XComet Score: {mean_score:.4f}")
           
def main() -> None:
  parser = transformers.HfArgumentParser(Arguments)
  (args,) = parser.parse_args_into_dataclasses()

  if torch.cuda.is_available():
    if args.model_size != "xxl":
      device = torch.device("cuda")
    else:
      device = torch.device("cuda:0")
    per_device_batch_size = args.batch_size // torch.cuda.device_count()
  else:
    device = torch.device("cpu")
    per_device_batch_size = args.batch_size
  print(f"Using model: {args.model_name}, size: {args.model_size}, device: {device}")
  tokenizer, model = get_tokenizer_and_model(args.model_name, args.model_size, args.dtype)
  if args.model_size != "xxl":
    model.to(device)
  
  # Check if we have language lists for batch processing
  if args.src_list is not None and args.tgt_list is not None:
    print("Batch processing mode: processing multiple language pairs")
    process_language_pairs(args.src_list, args.tgt_list, args, tokenizer, model, device, per_device_batch_size)
  else:
    print("Single language pair mode")
    process_single_language_pair(args, tokenizer, model, device, per_device_batch_size)

if __name__ == "__main__":
  main()

