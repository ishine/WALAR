import json
import tqdm
import os
import transformers
import datasets

from tqdm import tqdm
from openai import OpenAI

from utils import write_to_file
from mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer, TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING, validate_number
from mqm_utils import TEMPLATE_DA, extract_boxed_number

from collections import defaultdict
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, List, Dict

key = "sk-6a7eb14e1db5468884b8bf8761fb3e02"

client = OpenAI(api_key=key, base_url="https://api.deepseek.com/beta")

lang_dict = {
    'eng': "English",
    "zho_simpl": "Chinese",
    'swh': "Swahili",
    "tam": "Tamil",
    "fra": 'French',
    "deu": "German",
    "spa": "Spanish",
    "ben": "Bengali",
    "hin": "Hindi",
    "jpn": "Japanese",
    "tgl": "Filipino (Tagalog)",
    "fin": "Finnish",
    "ara": "Arabic",
    "tur": "Turkish",
    ### Indo-European-Slavic
    "bel": "Belarusian",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "hrv": "Croatian",
    "ces": "Czech",
    "mkd": "Macedonian",
    "pol": "Polish",
    "rus": "Russian",
    "srp": "Serbian",
    "slk": "Slovak",
    "sva": "Slovenian",
    "ukr": "Ukrainian",
}

@dataclass
class EvaluationArguments:
    """
    Arguments for model evaluation.
    """
    eval_type: str = field(
        default="mqm",
        metadata={"help": "Type of evaluation to perform (e.g., mqm, esa_error_spans, esa_ranking)"}
    )
    input_file: str = field(
        default="/dev/null",
        metadata={"help": "Directory containing FLORES-101 dataset"}
    )
    src: str = field(
        default="eng",
        metadata={"help": "Source language code (e.g., eng, zho_simpl, tam)"}
    )
    tgt: str = field(
        default="fra",
        metadata={"help": "Target language code (e.g., fra, deu, spa)"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save evaluation results"}
    )
    max_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens to generate"}
    )
    turns: int = field(
        default=1,
        metadata={"help": "Number of turns for the evaluation"}
    )
    
def my_load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(json.loads(line.strip()))
    return dataset

def preprocess_dataset(path):
  # Load with my own function because of potential error when loading low-resource languages
  name = ''
  if 'IndicMT' in path:
    name = 'IndicMT'
    ds = my_load_dataset(path)
    for data in ds:
      data['source'] = data.pop('src')
      data['hypothesis'] = data.pop('translation')
      data['reference'] = data.pop('ref')
  elif 'wmt' in path:
    name = 'wmt'
    with open(path, newline='') as f:
      reader = csv.DictReader(f, delimiter='\t')
      ds = []
      for row in reader:
        row['hypothesis'] = row.pop('target')
        ds.append(row)
  elif 'afriMTE' in path:
    name = 'afriMTE'
    ds = my_load_dataset(path)
    for data in ds:
      data['source'] = data.pop('src')
      data['hypothesis'] = data.pop('hypothesis')
      data['reference'] = data.pop('reference')
  else:
    raise ValueError(f"Unsupported dataset: {path}")
  return ds, name

def get_response(prompts):
    responses = []
    for idx in tqdm(range(len(prompts))):
        prompt = prompts[idx]
        # import code; code.interact(local=locals())
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=prompt,
            max_tokens=128,
            stop=None,
            presence_penalty=0,
            stream=False,
            temperature=1
        )
        # import code; code.interact(local=locals())
        responses.append([response.choices[0].message.content.strip()])
    # import code; code.interact(local=locals())
    return responses

def get_scores(eval_type: str, ds: List[Dict]):
    # source_lang, source_seg, target_lang, target_seg
    prompts = []
    for data in ds:
        temp_data = {
            'source_lang': data['src_lang'],
            'target_lang': data['tgt_lang'],
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
        
        prompts.append(prompt)
    outputs1 = get_response(prompts)
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
                'source_lang':  data['src_lang'],
                'target_lang':  data['tgt_lang'],
                'source_seg':   data['source'],
                'target_seg':   data['hypothesis'],
                "error_spans":  error_span,
            }
            prompt = apply_template(TEMPLATE_GEMBA_ESA_RANKING, temp_data)
            prompt = [{"role": "user", "content": prompt}]
            # prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(prompt)
        # outputs = model.generate(prompts, sampling_params=sampling_params)
        # outputs = [output.outputs[0].text for output in outputs]
        # scores  = [validate_number(output) for output in outputs]
        outputs = get_response(prompts)
        scores = [extract_boxed_number(output) for output in outputs]
        scores = [scores[i:i+turns] for i in range(0, len(scores), turns)]
    elif eval_type == "da":
        # scores = [extract_boxed_number(output) for output in outputs1]
        scores = [[extract_boxed_number(opt) for opt in output] for output in outputs1]
    
    # scores = [score if score is not None else 0 for score in scores]
    # import code; code.interact(local=locals())
    scores = [[0 if sc is None else sc for sc in s]for s in scores]
    scores = [sum(s) / len(s) for s in scores]
    # import code; code.interact(local=locals())
    return scores


if __name__ == '__main__':
    parser = transformers.HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(f"Evaluating model Deepseek...")
    ds, name = preprocess_dataset(args.input_file)
    # ds = ds[:1]
    ds = datasets.Dataset.from_list(ds)
    # ds structure: source, hypothesis, reference
    
    scores = get_scores(args.eval_type, ds)
    dirname = args.output_dir
    dirname = os.path.join(dirname)
    
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    output_file = os.path.join(
        dirname,
        f"{args.src}-{args.tgt}.jsonl",
    )
    write_to_file(output_file, ds, scores, "deepseek")
    