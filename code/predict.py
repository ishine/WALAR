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

sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
from utils import write_to_file, preprocess_dataset, my_load_dataset
import models
import datasets
from typing import Optional, Tuple, Union, List
# from code import models
import torch
import transformers
from transformers import DataCollatorWithPadding


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
    model_output = model.predict(inputs, batch_size=per_device_batch_size, gpus=torch.cuda.device_count())
    predictions = model_output.scores
  else:
    raise ValueError("Unsupported model name or path: {}".format(model_name))
  
  return predictions

           
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
  #     "hypothesis": "加拿大糖尿病协会临床与科研分会主席、达尔豪斯大学哈利法克斯分校医学院教授埃胡德·厄尔博士指出，目前这项研究仍处于初步阶段，尚需进一步深入探讨。",
  #     "reference": "埃胡德·乌尔博士（新斯科舍省哈利法克斯市达尔豪西大学医学教授，加拿大糖尿病协会临床与科学部门教授）提醒，这项研究仍处在早期阶段。",
  #   }
  # ]
  ds = [
      {
          "source": "Dr. Tony Moll discovered the Extremely Drug Resistant Tuberculosis (XDR-TB) in the South African region KwaZulu-Natal.",
          "hypothesis": "Dr. Tony Moll在南非KwaZulu-Natal地区发现了一种非常难治疗的结核病类型——Extremely Drug Resistant Tuberculosis（XDR-TB）。这种病菌对大多数常规抗生素治疗无效，需要使用特定的抗结核药物进行治疗。",
          "reference": "托尼·莫尔博士在南非夸祖鲁-纳塔尔省发现了这种广泛耐药结核病 (XDR-TB)。",
      }
  ]
  # ds = [
  #   {
  #     "source": "Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features.",
  #     "hypothesis": "由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。",
  #     "reference": "恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。",
  #   }
  # ]
  ds = datasets.Dataset.from_list(ds)
  ds, data_collator = get_dataset(
      ds,
      args.model_name,
      tokenizer,
      args.max_input_length,
      device,
      args.qe,
  )
  predictions = get_predictions(ds, model, data_collator, per_device_batch_size, args.model_name, args.output_dir)
  # print(predictions[0])
  dirname = args.output_dir
  dirname = os.path.join(dirname, args.model_name + "-" + args.model_size + "-" + args.dtype)
  import code; code.interact(local=locals())
  # if dirname:
  #   os.makedirs(dirname, exist_ok=True)

  # output_file = os.path.join(
  #     dirname,
  #     f"{args.src}-{args.tgt}.jsonl",
  # )
  # write_to_file(output_file, ds, predictions, args.model_name)


if __name__ == "__main__":
  main()

