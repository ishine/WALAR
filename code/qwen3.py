import json
import csv
import transformers
from transformers import AutoTokenizer
import argparse
import datasets
import openai
import os
import matplotlib.pyplot as plt
from utils import preprocess_dataset, mm_dict, lang_dict
from mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer, TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING, validate_number
from mqm_utils import TEMPLATE_DA, extract_boxed_number

from vllm import LLM, SamplingParams
from collections import defaultdict
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class EvaluationArguments:
    """
    Arguments for model evaluation.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
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
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs to use for tensor parallelism"}
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

def write_to_file(output_file, ds, predictions):
  with open(output_file, "w") as out:
    for pred, example in zip(predictions, ds):
      example["prediction"] = float(pred) if pred is not None else pred
      out.write(json.dumps(example) + "\n")

def load_flores_dataset(data_dir, lang_pair):
    """Load FLORES-101 dataset for a specific language pair."""
    # dataset = load_dataset("facebook/flores", "all")
    def my_load_dataset(data_pair, lang):
        dataset = []
        path = os.path.join(data_pair, f"{lang}.devtest")
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(line.strip())
        return dataset
    src_lang, tgt_lang = lang_pair.split("-")
    
    # Get test split
    src_dataset, tgt_dataset = my_load_dataset(data_dir, src_lang), my_load_dataset(data_dir, tgt_lang)
    
    # Filter for the specific language pair
    return src_dataset, tgt_dataset

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
    # import code; code.interact(local=locals())
    scores = [[0 if sc is None else sc for sc in s]for s in scores]
    scores = [sum(s) / len(s) for s in scores]
    # import code; code.interact(local=locals())
    return scores

def get_langs(args):
    src, tgt = args.src, args.tgt
    src_lang, tgt_lang = mm_dict.get(src, ''), mm_dict.get(tgt, '')
    if len(src_lang) == 0 or len(tgt_lang) == 0:
        src_lang, tgt_lang = lang_dict.get(src, ''), lang_dict.get(tgt, '')
    # The case for IndicMT
    if tgt_lang == '':
        tgt_lang = args.tgt.capitalize()
        # raise ValueError(f"Unsupported language codes: {src}, {tgt}")
    print(f"Source language: {src_lang}, Target language: {tgt_lang}")
    # import code; code.interact(local=locals())
    return src_lang, tgt_lang

def main():
    parser = transformers.HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(f"Evaluating model {args.model_name_or_path}...")
    # ds, name = preprocess_dataset(args.input_file)
    # ds = datasets.Dataset.from_list(ds)
    # dir_path = "/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"
    # src_dataset, tgt_dataset = load_flores_dataset(dir_path, f"eng-uzb")
    # ds = [{
    #     "src_lang": "English",
    #     "tgt_lang": "Serbian",
    #     "source": src,
    #     "hypothesis": tgt
    # } for src, tgt in zip(src_dataset, tgt_dataset)]
    ds = [
        {
            "src_lang": "English",
            "tgt_lang": "German",
            "source": 'The other nominations include Best Picture, Director, Cinematography, Costume Design, Film-editing, Original Score, Production Design, Sound Editing, Sound Mixing and Original Screenplay.',
            "hypothesis": 'Die anderen Nominierungen umfassen Best Picture, Director, Cinematography, Costume Design, Film-editing, Original Score, Production Design, Sound Editing, Sound Mixing und Original Screenplay.'
        }
    ]
    # ds structure: source, hypothesis, reference
    if args.model_name_or_path == "Qwen3-235B-AWQ":
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5, max_tokens=args.max_tokens, n=args.turns)
    elif args.model_name_or_path == "Qwen3-32B-AWQ" or args.model_name_or_path == "Qwen3-30B-A3B" or args.model_name_or_path == "Qwen3-235B-Instruct":
        sampling_params = SamplingParams(temperature=1, top_p=0.9, top_k=-1, min_p=0, max_tokens=args.max_tokens, n=args.turns)
    else:
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=args.max_tokens, n=args.turns)
        # sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=-1, presence_penalty=0, frequency_penalty=0, max_tokens=args.max_tokens, n=args.turns)

    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size, task="generate", enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    src_lang, tgt_lang = get_langs(args)
    scores = get_scores(args.eval_type, ds, sampling_params, model, tokenizer, src_lang, tgt_lang)
    dirname = args.output_dir
    dirname = os.path.join(dirname)
    
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    output_file = os.path.join(
        dirname,
        f"{args.src}-{args.tgt}.jsonl",
    )
    write_to_file(output_file, ds, scores)
    
    # plt.hist(scores, bins=[i for i in range(101)], edgecolor='black')

    # plt.xlabel('Value Range')
    # plt.ylabel('Count')
    # plt.title('Score Distribution')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # # plt.show()
    # plt.savefig(f"/mnt/gemini/data1/yifengliu/qe-lr/output/eng-uzb.png")
    # print(f"Align Score: {scores}")

if __name__ == "__main__":
    main()