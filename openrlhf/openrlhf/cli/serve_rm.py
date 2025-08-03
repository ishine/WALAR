import dataclasses
import argparse
import os
import sys
import re
import jieba

sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
import models
from sentence_transformers import SentenceTransformer
from typing import Any, List, Tuple, Union, Optional
from tqdm import *

import torch
import transformers
import datasets
import uvicorn
import fasttext
import sacrebleu
import itertools
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


def align_score(srcs, tgts, model, tokenizer):
  align_score_list = []
  # for src, tgt in zip(srcs, tgts):
  for i in tqdm(range(len(srcs))):
    sent_src, sent_tgt = srcs[i], tgts[i]
    # sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    # print(token_src)
    # print(token_tgt)
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
      sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
      sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
      out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
      out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

      dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

      softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
      softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

      softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
      align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
    align_percent = len(align_words) / len(token_tgt)
    align_score_list.append(align_percent)
  return align_score_list

class RewardModelProxy:
    def __init__(self, args):
        self.args = args
        self.base_model = args.base_model
        if args.lang_detect:
          self.lang_detect_model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
        if args.align:
          model_path = "/mnt/gemini/data1/yifengliu/model/LaBSE"
          # self.align_model = SentenceTransformer(model_path)
          self.align_model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
          self.align_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
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
        logger.info(f"queries[1]: {queries[1]}")

        scores = []
        # batch
        if "metricX" in self.model_name:
          with torch.no_grad():
              ds = []
              src_pattern = r"<\|im_start\|>user\n(.*?)Translate from (.*?) to (.*?)"
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
              predictions, _, _ = self.trainer.predict(test_dataset=dataset)
              scores.extend(-predictions)
        elif 'Comet' in self.model_name:
          ds = []
          src_pattern = r"<\|im_start\|>user\n(.*?)Translate from (.*?) to (.*?)"
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
          min_reward = -25 if 'metricX' in self.model_name else 0
          new_scores = []
          cnt = 0
          for score, tgt in zip(scores, tgts):
            # print(tgt, '\n' in tgt)
            if "\n" in tgt:
              cnt += 1
              new_scores.append(min_reward)
            else:
              new_scores.append(score)
          scores = new_scores
          extra_logs['rule_penalty_percent'] = cnt / len(tgts)
        
        if self.args.lang_detect:
          pattern = r"Translate from English to ([^\n<]+):"
          target_languages = [re.search(pattern, query).group(1).strip() for query in queries if re.search(pattern, query)]
          tgts = [tgt.replace("\n", "") for tgt in tgts]
          lang_info = self.lang_detect_model.predict(tgts)
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
          scores = [min(score, detect_reward) for score, detect_reward in zip(scores, detect_rewards)]
          logger.info(lang_info[0][:20])
          extra_logs['lang_penalty_percent'] = cnt / len(tgts)
          
        if self.args.truncate:
          truncate_bound = -3
          extra_logs['truncate_percent'] = sum(score >= truncate_bound for score in scores) / len(scores)
          scores = [score if score < truncate_bound else truncate_bound for score in scores]
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
          # align_scores = []
          # for tgt, label in zip(tgts, labels):
          #   src_embedding = self.align_model.encode(tgt)
          #   tgt_embedding = self.align_model.encode(label)
          #   similarity = src_embedding @ tgt_embedding.T
          #   align_scores.append(float(similarity))
          # scores = [score + 100*align_score for score, align_score in zip(scores, align_scores)]
          # extra_logs['mean_align_score'] = sum(align_scores) / len(align_scores)
          print(srcs[0])
          print(tgts[0])
          # srcs = [self.align_tokenizer.tokenize(src) for src in srcs]
          # srcs = [" ".join(src) for src in srcs]
          srcs = [src.strip().split() for src in srcs]
          # tgts = [self.align_tokenizer.tokenize(tgt) for tgt in tgts]
          tgts = [list(jieba.cut(tgt.strip())) for tgt in tgts]
          tgts = [[t for t in tgt if len(t.strip()) > 0] for tgt in tgts]
          # tgts = [" ".join(tgt) for tgt in tgts]
          align_score_list = align_score(srcs, tgts, self.align_model, self.align_tokenizer)
          align_score_list = [score*6 for score in align_score_list]
          print(align_score_list[:20])
          scores = [score + align_score for score, align_score in zip(scores, align_score_list)]
          extra_logs['mean_align_score'] = sum(align_score_list) / len(align_score_list)
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
