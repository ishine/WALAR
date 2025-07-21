import argparse
import os
import sys
import re

sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
from mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer, TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING, validate_number
from mqm_utils import TEMPLATE_DA, extract_boxed_number
from typing import Any, List, Tuple, Union


import torch
import transformers
import datasets
import uvicorn
import fasttext
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import DataCollatorWithPadding, AutoTokenizer
from vllm import LLM, SamplingParams
from typing import Optional, List, Dict
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

def get_scores(eval_type: str, ds: List[Dict], sampling_params: SamplingParams, model: LLM, tokenizer: AutoTokenizer=None, src_lang_list: List[str]="English", tgt_lang_list: List[str]="Chinese"):
    # source_lang, source_seg, target_lang, target_seg
    prompts = []
    for data, src_lang, tgt_lang in zip(ds, src_lang_list, tgt_lang_list):
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

    outputs1 = [[opt.text for opt in output.outputs] for output in outputs]
    if eval_type == "mqm":
        scores = [[parse_mqm_answer(opt) for opt in output] for output in outputs1]
    elif eval_type == "esa":
        print("Evaluating ESA Error Spans...")
        prompts = []
        n1 = len(outputs1)
        outputs1 = [opt for output in outputs1 for opt in output]
        turns = len(outputs1) // n1
        for data, error_span, src_lang, tgt_lang in zip(ds, outputs1, src_lang_list, tgt_lang_list):
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

def get_sampling_params(metric_name, args):
  if metric_name == "Qwen3-235B":
      sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5, max_tokens=2048, n=args.turns)
  elif metric_name == "Qwen3-32B-AWQ":
      sampling_params = SamplingParams(temperature=1, top_p=0.9, top_k=-1, min_p=0, max_tokens=2048, n=args.turns)
  else:
      # sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=args.max_tokens, n=args.turns)
      sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=-1, presence_penalty=0, frequency_penalty=0, max_tokens=2048, n=args.turns)
  return sampling_params

class RewardModelProxy:
    def __init__(self, args):
        self.args = args
        self.src = args.src
        self.tgt = args.tgt
        # self.lang_detect_model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
        self.model_name = args.model_name
        self.base_model = args.base_model
        path_dict = {
          "Qwen3-32B": "/mnt/gemini/data1/yifengliu/model/Qwen3-32B",
          "Qwen3-32B-AWQ": "/mnt/gemini/data1/yifengliu/model/Qwen3-32B-AWQ",
          "Qwen3-235B": "/mnt/gemini/data1/yifengliu/model/Qwen3-235B-A22B-GPTQ-Int4",
        }
        self.model_path = path_dict.get(self.model_name, None)
        self.model = LLM(model=self.model_path, tensor_parallel_size=args.tensor_parallel_size, task="generate", enforce_eager=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.sampling_params = get_sampling_params(self.model_name, args)
        # self.model = 

    def get_reward(self, queries, prompts, labels):

      logger.info(f"queries[0]: {queries[0]}")
      logger.info(f"queries[1]: {queries[1]}")

      scores = []
      # batch
      src_pattern = r"<\|im_start\|>user\n(.*?)Translate from (.*?) to (.*?)"
      srcs = [re.search(src_pattern, q, re.DOTALL).group(1).strip() for q in queries]
      src_langs = [re.search(src_pattern, q, re.DOTALL).group(2).strip() for q in queries]
      tgt_langs = [re.search(src_pattern, q, re.DOTALL).group(3).strip() for q in queries]

      # Match tgt between "<|im_start|>assistant\n" and "<|im_end|>"
      # tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>(.*?)<\|im_end\|>"
      print(f"queries[0]: {queries[0]}")
      if 'Qwen3' in self.base_model:
        tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>\n\n(.*?)<\|im_end\|>"
        tgts = [re.search(tgt_pattern, q, re.DOTALL).group(2).strip() for q in queries]
      else:
        tgt_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
        tgts = [re.search(tgt_pattern, q, re.DOTALL).group(1).strip() for q in queries]
      ds = [{"source": src, "hypothesis": tgt} for src, tgt in zip(srcs, tgts)]
      scores = get_scores(args.eval_type, ds, self.sampling_params, self.model, self.tokenizer, src_langs, tgt_langs)
      # print(f"{self.model_name}: query: {queries[0]}")
      # print(f"{self.model_name}: prompt: {prompts[0]}")
      # print(f"{self.model_name}: score: {scores[0]}")
      extra_logs = {}
      return scores, extra_logs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument('--model_name', type=str, default="metricX")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    parser.add_argument('--base_model', type=str, default="Qwen2.5-3B-Instruct", help="Base model name or path")
    parser.add_argument("--eval_type", type=str, default="mqm", choices=["mqm", "esa", "da"], help="Evaluation type: mqm, esa, or da")
    parser.add_argument("--src", type=str, default="en", help="Source language code")
    parser.add_argument("--tgt", type=str, default="zh", help="Target language code")
    parser.add_argument("--lang_detect", type=bool, default=False, help="Enable language detection")
    parser.add_argument("--rule", type=bool, default=False, help="Rule to use \\n as a reward or not")
    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for distributed training")
    parser.add_argument("--turns", type=int, default=1, help="Number of turns for the model to generate")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--packing_samples", action="store_true", default=False)

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
        logger.info(f"Sent JSON: {result['rewards'][:20]}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

