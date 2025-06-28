import argparse
import os
import sys

sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
import models
from typing import Any, List, Tuple, Union


import torch
import transformers
import datasets
import uvicorn
import fasttext
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import DataCollatorWithPadding

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

lang_dict = {
  "en": "English",
  "zh": "Chinese",
  "sw": "Swahili",
}


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


class RewardModelProxy:
    def __init__(self, args):
        self.args = args
        self.src = args.src
        self.tgt = args.tgt
        self.lang_detect_model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
        if 'metricX' in args.model_name:
            self.model_name = args.model_name
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl", cache_dir="/mnt/gemini/data1/yifengliu/model")
            self.model = models.MT5ForRegression.from_pretrained(
                "google/metricx-24-hybrid-xxl-v2p6-bfloat16", torch_dtype="auto", device_map="auto", cache_dir="/mnt/gemini/data1/yifengliu/model"
            )
            self.max_length = args.max_len
            self.batch_size = args.batch_size
            self.training_args = transformers.TrainingArguments(
                output_dir="/mnt/gemini/data1/yifengliu/qe-lr/output/openrlhf",
                per_device_eval_batch_size=self.batch_size//torch.cuda.device_count(),
                dataloader_pin_memory=False,
            )
            self.trainer = transformers.Trainer(
                model=self.model,
                args=self.training_args,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True),
            )
        elif 'Comet' in args.model_name:
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

        scores = []
        # batch
        if "metricX" in self.model_name:
          with torch.no_grad():
              ds = []
              srcs = [query.split('<|im_start|>user\n', 1)[1].split(f"Translate from {lang_dict[self.src]} to {lang_dict[self.tgt]}", 1)[0].strip() for query in queries]
              tgts = [query.split('<|im_start|>assistant\n', 1)[1].split("<|im_end|>", 1)[0].strip() for query in queries]
              # srcs = [query.split('user\n', 1)[1].split("Translate from English to Chinese", 1)[0].strip() for query in queries]
              # tgts = [query.split('Translate from English to Chinese:\nassistant\n', 1)[1] for query in queries]
              for src, tgt, label in zip(srcs, tgts, labels):
                  ds.append({"source": src, "hypothesis": tgt, 'reference': label})
              ds = datasets.Dataset.from_list(ds)
              if 'ref' in self.model_name:
                dataset = get_dataset(ds, self.model_name, self.tokenizer, self.max_length, self.model.device, is_qe=False)
              else:
                dataset = get_dataset(ds, self.model_name, self.tokenizer, self.max_length, self.model.device, is_qe=True)
              # import code; code.interact(local=locals())
              # print(dataset)
              predictions, _, _ = self.trainer.predict(test_dataset=dataset)
              scores.extend(-predictions)
        elif 'Comet' in self.model_name:
          ds = []
          srcs = [query.split('<|im_start|>user\n', 1)[1].split(f"Translate from {lang_dict[self.src]} to {lang_dict[self.tgt]}", 1)[0].strip() for query in queries]
          tgts = [query.split('<|im_start|>assistant\n', 1)[1].split("<|im_end|>", 1)[0].strip() for query in queries]
          inputs = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, tgts, labels)]
    
          output = self.model.predict(inputs, batch_size=8, gpus=1)
          mean_score = output.system_score
          scores.extend(output.scores)
          # scores, mean_score = output.scores, output.system_score
        # print(f"{self.model_name}: query: {queries[0]}")
        # print(f"{self.model_name}: prompt: {prompts[0]}")
        # print(f"{self.model_name}: score: {scores[0]}")
        extra_logs = {}
        if self.args.rule:
          min_reward = -25 if 'metricX' in self.model_name else 0
          new_scores = []
          cnt = 0
          for score, tgt in zip(scores, tgts):
            if "\n" in tgt:
              cnt += 1
              new_scores.append(min_reward)
            else:
              new_scores.append(score)
          scores = new_scores
          extra_logs['rule_penalty_percent'] = cnt / len(tgts)
        
        if self.args.lang_detect:
          tgts = [tgt.replace("\n", "") for tgt in tgts]
          lang_info = self.lang_detect_model.predict(tgts)
          min_reward = -25 if 'metricX' in self.model_name else 0
          detect_rewards = []
          cnt = 0
          for language in lang_info[0]:
            if language[0].replace("__label__", "") == self.tgt:
              detect_rewards.append(float('inf'))
            else:
              cnt += 1
              detect_rewards.append(min_reward)
          scores = [min(score, detect_reward) for score, detect_reward in zip(scores, detect_rewards)]
          logger.info(lang_info[0][:50])
          extra_logs['lang_penalty_percent'] = cnt / len(tgts)
        return scores, extra_logs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument('--model_name', type=str, default="metricX")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    parser.add_argument("--src", type=str, default="en", help="Source language code")
    parser.add_argument("--tgt", type=str, default="zh", help="Target language code")
    parser.add_argument("--lang_detect", type=bool, default=False, help="Enable language detection")
    parser.add_argument("--rule", type=bool, default=False, help="Rule to use \\n as a reward or not")
    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--packing_samples", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

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
        logger.info(f"Sent JSON: {result['rewards'][:50]}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
