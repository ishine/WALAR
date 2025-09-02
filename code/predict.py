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
from utils import write_to_file, preprocess_dataset, my_load_dataset
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
    alignment: bool = False,
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
  model.eval()
  ds, name = preprocess_dataset(args.input_file)
  # ds = [
  #   {
  #     "source": "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.", 
  #     "hypothesis": "Dr. Ehud Ur，Dalhousie University 在 Halifax，Nova Scotia 的医学教授，以及 Canadian Diabetes Association 的临床和科学分部主席，提醒说这项研究仍处于早期阶段。"
  #   }
  # ]
  # ds = [
  #   {
  #     "source": "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.",
  #     "hypothesis": "On Monday, scientists from the Stanford University School of Medicine announced the development of a new diagnostic tool. This tool can identify cells based on their type. It is a tiny, printable chip that can be produced using standard inkjet printers. The cost of this tool is approximately one U.S. cent each."
  #   }
  # ]
  # -2.88
  # ds = [
  #   {
  #     "source": "\"We now have 4-month-old mice that are non-diabetic that used to be diabetic,\" he added.",
  #     "hypothesis": "我们目前有4个月大的小鼠，这些小鼠都是非糖尿病状态，而以前的糖尿病小鼠则都患有糖尿病。他进一步解释说：“这些小鼠在实验过程中没有受到任何药物或营养物质的干扰，它们的生理状态和代谢过程都保持正常。这表明这些小鼠的健康状况非常良好，没有受到任何有害因素的影响。我们可以通过这些小鼠来研究糖尿病的病因、治疗方案以及预防措施等，为人类的健康研究提供宝贵的资料。”\n中文翻译：\n我们目前有4个月大的小鼠，这些小鼠都是非糖尿病状态，而以前的糖尿病小鼠则都患有糖尿病。他进一步解释说：“这些小鼠在实验过程中没有受到任何药物或营养物质的干扰，它们的生理状态和代谢过程都保持正常。这表明这些小鼠的健康状况非常良好，没有受到任何有害因素的影响。我们可以通过这些小鼠来研究糖尿病的病因、治疗方案以及预防措施等，为人类的健康研究提供宝贵的资料。”"
  #   }
  # ]
  # ds = [
  #   {
  #     "source": "On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.",
  #     "hypothesis": "周一，萨拉·丹努斯（Sara Danius），瑞典学院（Swedish Academy）文学奖委员会的常任秘书，在瑞典广播电台（Sveriges Radio）的一次广播节目中公开宣布，由于无法直接联系到鲍勃·迪伦（Bob Dylan）关于获得2016年诺贝尔文学奖一事，委员会已放弃了尝试联系他的努力。",
  #   }
  # ]
  # ds = [
  #   {
  #     "source": "Danius said, \"Right now we are doing nothing. I have called and sent emails to his closest collaborator and received very friendly replies. For now, that is certainly enough.\"",
  #     "hypothesis": "Danius说，“现在我们正在做 nothing。我已经打电话并发送电子邮件给他的最亲近的合作者，并收到了非常友好的回复。目前来说，这已经足够了。”"
  #   }
  # ]
  # ds = [
  #   {
  #     "source": "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.",
  #     "hypothesis": "加拿大糖尿病协会（Canadian Diabetes Association, Kanadischen Diabetesverbands）临床与科学分会主席、达尔豪斯大学（Dalhousie University）哈利法克斯（Halifax, Nova Scotia）医学院教授埃胡德·乌（Ehud Ur）警告说，这项研究仍处于早期阶段。"
  #   }
  # ]
  # ds = [
  #   {
  #     "source": 'Current senator and Argentine First Lady Cristina Fernandez de Kirchner announced her presidential candidacy yesterday evening in La Plata, a city 50 kilometers (31 miles) away from Buenos Aires.',
  #     "hypothesis": "现任参议员及阿根廷第一夫人克ristina Fernández de Kirchner于昨晚在拉普拉塔（La Plata），这座距离布宜诺斯艾利斯（Buenos Aires）约50公里（31英里）的城市宣布参选总统"
  #   }
  # ]
  # ds = [
  #   {
  #     "source": "The other nominations include Best Picture, Director, Cinematography, Costume Design, Film-editing, Original Score, Production Design, Sound Editing, Sound Mixing and Original Screenplay.",
  #     "hypothesis": 'Die anderen Nominierungen umfassen Best Picture, Director, Cinematography, Costume Design, Film-editing, Original Score, Production Design, Sound Editing, Sound Mixing und Original Screenplay.'
  #   }
  # ]
  # ds = [
  #   {
  #     "source": "On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.",
  #     "hypothesis": "周一，瑞典学院文学奖委员会永久秘书萨拉·丹努斯在瑞典广播电台的一档节目中宣布，由于无法直接联系到鲍勃·迪伦，委员会放弃了尝试联系他的努力。"
  #   }
  # ]
  # ds = [
  #   {
  #     "source": "The other nominations include Best Picture, Director, Cinematography, Costume Design, Film-editing, Original Score, Production Design, Sound Editing, Sound Mixing and Original Screenplay. ",
  #     "hypothesis": "Die anderen Nominierungen umfassen Best Picture, Director, Cinematography, Costume Design, Film-editing, Original Score, Production Design, Sound Editing, Sound Mixing und Original Screenplay.",
  #   }
  # ]
  # ds = [
  #   {
  #     "source": "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.",
  #     "hypothesis": "Dr. Ehud Ur, dosen kedokteran di Universitas Dalhousie di Halifax, Nova Scotia, dan ketua divisi klinis dan ilmiah Asosiasi Diabetes Kanada memperingatkan bahwa penelitian ini masih dalam tahap awal.",
  #     "reference": "Dr. Ehud Ur, profesor ilmu kedokteran ing Universitas Dalhousie ing Halifax, Nova Scotia lan ketua divisi klinis lan ilmiah saka Asosiasi Diabetes Kanada ngengetake menawa panaliten iku isih ing tahap wiwitan.",
  #   }
  # ]
  # src_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/eng.devtest"
  # tgt_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/azj.devtest"
  # src_dataset, tgt_dataset = load_flores(src_path), load_flores(tgt_path)
  # ds = [{
  #   "source": src,
  #   "hypothesis": tgt,
  # } for src, tgt in zip(src_dataset, tgt_dataset)]
  dt = datasets.Dataset.from_list(ds)
  dt, data_collator = get_dataset(
      dt,
      args.model_name,
      tokenizer,
      args.max_input_length,
      device,
      args.qe,
  )
  predictions = get_predictions(dt, model, data_collator, per_device_batch_size, args.model_name, args.output_dir, args.alignment)
  # predictions = [0]*len(predictions)
  if args.alignment:
    align_path = "/mnt/gemini/data1/yifengliu/model/bge-m3"
    align_model = AutoModel.from_pretrained(align_path)
    align_tokenizer = AutoTokenizer.from_pretrained(align_path)
    align_model.to("cuda:0")
    srcs, tgts = [d['source'] for d in ds], [d['hypothesis'] for d in ds]
    src_copy = [src.split() for src in srcs]
    tgt_copy = [tgt.split() for tgt in tgts]
    scores = align_score(src_copy, tgt_copy, align_model, align_tokenizer, batch_size=16)
    predictions = [prediction + score for prediction, score in zip(predictions, scores)]
  if args.lang:
    srcs, tgts = [d['source'] for d in ds], [d['hypothesis'] for d in ds]
    lang_detect_model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
    lang_info = lang_detect_model.predict(tgts)
    lang_id = [language[0].replace("__label__", "") for language in lang_info[0]]
    # pred_lang = [lang_dict.get(lang, 0) for lang in lang_id]
    new_score_list = [-25 if lang != args.tgt else float("inf") for lang in lang_id]
    # import code; code.interact(local=locals())
    predictions = [min(prediction, score) for prediction, score in zip(predictions, new_score_list)]
  # print(predictions[0])
  dirname = args.output_dir
  if not args.alignment:
    dirname = os.path.join(dirname, args.model_name + "-" + args.model_size + "-" + args.dtype)
  else:
    dirname = os.path.join(dirname, args.model_name + "-" + args.model_size + "-" + args.dtype + "-align")
  # import code; code.interact(local=locals())
  if name != "flores":
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
      if args.model_name == "XComet":
        f.write(f"XComet Score: {mean_score:.4f}\n")
        print(f"{args.src}-{args.tgt}: XComet Score: {mean_score:.4f}")

if __name__ == "__main__":
  main()

