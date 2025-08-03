import os
import datasets
import numpy as np
import torch
import transformers
import json
import sacrebleu

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from utils import lang_dict

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
from typing import Optional
from sacrebleu.metrics import BLEU, CHRF
from comet import load_from_checkpoint, download_model

@dataclass
class EvaluationArguments:
    """
    Arguments for model evaluation.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lang_pair: str = field(
        default="eng-hin",
        metadata={"help": "Language pair for evaluation (e.g., eng-hin)"}
    )
    data_dir: str = field(
        default="/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest",
        metadata={"help": "Directory containing FLORES-101 dataset"}
    )
    output_file: Optional[str] = field(
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
    comet22: bool = field(
        default=False,
        metadata={"help": "Whether to compute COMET22 score"}
    )
    xcomet: bool = field(
        default=False,
        metadata={"help": "Whether to compute XCOMET score"}
    )


def my_load_dataset(data_pair, lang):
    dataset = []
    path = os.path.join(data_pair, f"{lang}.devtest")
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(line.strip())
    return dataset

def get_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    hyps = [hyp.strip() for hyp in hyps]
    refs = [ref.strip() for ref in refs]
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="spm", force=True).score
    return result

def load_flores_dataset(data_dir, lang_pair):
    """Load FLORES-101 dataset for a specific language pair."""
    # dataset = load_dataset("facebook/flores", "all")
    src_lang, tgt_lang = lang_pair.split("-")
    
    # Get test split
    src_dataset, tgt_dataset = my_load_dataset(data_dir, src_lang), my_load_dataset(data_dir, tgt_lang)
    
    # Filter for the specific language pair
    return src_dataset, tgt_dataset

def predict(model, tokenizer, dataset, sampling_params, lang_pair, model_path):
    """Generate predictions using the model."""
    src_lang, tgt_lang = lang_pair.split("-")
    src_lang, tgt_lang = lang_dict[src_lang], lang_dict[tgt_lang]
    prompts = []
    if "nllb" not in model_path:
        for src_text in tqdm(dataset, desc="Generating predictions"):
            # <X> \n Translate from [SRC] to [TGT]: \n <Y>
            prompt = f"{src_text}\nTranslate from {src_lang} to {tgt_lang}:\n"
            message = [
                {"role": "user", "content": prompt},
            ]
            new_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(new_prompt)
        responses = model.generate(prompts, sampling_params=sampling_params)
        responses = [response.outputs[0].text for response in responses]
    else:
        batch_size = 32
        responses = []
        for idx in tqdm(range(0, len(dataset), batch_size), desc="Generating predictions"):
            right_bound = min(idx + batch_size, len(dataset))
            sources = dataset[idx:right_bound]
            inputs = tokenizer(sources, return_tensors="pt", padding=True)
            inputs.to("cuda:0")
            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("deu_Latn"), max_length=512
            )
            output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True, model_max_length=512)
            responses.extend(output)
    return responses

def evaluate_model(model_path, data_dir, lang_pair, tensor_parallel_size=1, max_tokens=512):
    """Evaluate a model on FLORES-101 dataset using vLLM."""
    # Load model and tokenizer
    if "nllb" not in model_path:
        model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, token=True, src_lang="eng_Latn"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=True)
        model.to("cuda:0")
    
    # Load dataset
    src_dataset, tgt_dataset = load_flores_dataset(data_dir, lang_pair)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # Use greedy decoding
        top_p=1.0,
        top_k=-1,
        seed=0,
    )
    
    predictions = predict(model, tokenizer, src_dataset, sampling_params, lang_pair, model_path)
    
    
    return src_dataset, tgt_dataset, predictions

def calculate_metrics(references, predictions):
    """Calculate evaluation metrics."""
    sp_bleu_score = get_spBLEU(predictions, references)
    return sp_bleu_score

def calculate_comet_score(src_texts, references, predictions, model_path="/mnt/gemini/data1/yifengliu/model/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"):
    """Calculate COMET score."""
    model = load_from_checkpoint(model_path)
    
    # Prepare inputs for COMET
    inputs = [{"src": src.strip(), "mt": mt.strip(), "ref": ref.strip()} for src, mt, ref in zip(src_texts, predictions, references)]
    
    output = model.predict(inputs)
    
    scores, mean_score = output.scores, output.system_score
    return {"mean_score": mean_score, "scores": scores}

def main():
    parser = transformers.HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    print(f"Evaluating model {args.model_name_or_path} on {args.lang_pair}...")
    pred = []
    
    sources, references, predictions = evaluate_model(
        args.model_name_or_path, 
        args.data_dir,
        args.lang_pair, 
        args.tensor_parallel_size,
        args.max_tokens
    )
    pred.append(predictions)
    metrics = get_spBLEU(predictions, references)
    if args.comet22:
        comet_score = calculate_comet_score(
            sources, references, predictions
        )
        print(f"COMET22 Score: {comet_score['mean_score']:.4f}")
    if args.xcomet:
        xcomet_score = calculate_comet_score(
            sources, references, predictions,
            model_path="/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
        )
        print(f"XCOMET Score: {xcomet_score['mean_score']:.4f}")
    print("=====================================")
    print(f"args.model_name_or_path: {args.model_name_or_path}")
    print(f"Results for {args.lang_pair}:")
    print(f"spBLEU: {metrics:.4f}")
    print(f"COMET Score: {comet_score['mean_score']:.4f}")
        
    print(f"source: {sources[0]}")
    print(f"prediction: {predictions[0]}")
    print(f"reference: {references[0]}")
    # import code; code.interact(local=locals())

    dirname = os.path.dirname(args.output_file) if args.output_file else None
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for src, pred, ref in zip(sources, predictions, references):
                f.write(json.dumps({'src': src, 'pred': pred, 'ref': ref}, ensure_ascii=False) + '\n')
            f.write(f"spBLEU: {metrics:.4f}\n")
            if args.comet22:
                f.write(f"COMET Score: {comet_score['mean_score']:.4f}\n")
            if args.xcomet:
                f.write(f"XCOMET Score: {xcomet_score['mean_score']:.4f}\n")

if __name__ == "__main__":
    main()
