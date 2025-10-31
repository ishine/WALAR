import os
import datasets
import numpy as np
import torch
import transformers
import json
import sacrebleu
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from utils import lang_dict, lang2long

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
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
    lang_pair: Optional[str] = field(
        default=None,
        metadata={"help": "Language pair for evaluation (e.g., eng-hin). If provided, overrides source_languages and target_languages."}
    )
    enable_thinking: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to enable <think> in prompts for models that support it."}
    )
    source_languages: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of source languages (e.g., 'eng,deu,fra')"}
    )
    target_languages: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target languages (e.g., 'hin,ben,tam')"}
    )
    data_dir: str = field(
        default="/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest",
        metadata={"help": "Directory containing FLORES-101 dataset"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save evaluation results for all language pairs"}
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save evaluation results for single language pair"}
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

def predict(enable_thinking, model, tokenizer, dataset, sampling_params, lang_pair, model_path):
    """Generate predictions using the model."""
    src_lang, tgt_lang = lang_pair.split("-")
    src_lang, tgt_lang = lang_dict[src_lang], lang_dict[tgt_lang]
    prompts = []
    model_path = model_path.lower()
    if "nllb" not in model_path:
        for src_text in tqdm(dataset, desc="Generating predictions"):
            # <X> \n Translate from [SRC] to [TGT]: \n <Y>
            if 'llamax' in model_path:
                prompt = f"""Translate the following sentences from {src_lang} to {tgt_lang}.\n### Input:\n{src_text}\n"""
                # prompt = f"{src_text.strip()}\nTranslate from {src_lang} to {tgt_lang}:\n"
            else:
                prompt = f"{src_text.strip()}\nTranslate from {src_lang} to {tgt_lang}:\n"
                # prompt = f'Translate the following text from {src_lang} to {tgt_lang}.\n\n{src_lang} source:\n{src_text}\n\n{tgt_lang} translation:'
            message = [
                {"role": "user", "content": prompt},
            ]
            new_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            prompts.append({"prompt": new_prompt})
        # import code; code.interact(local=locals())
        responses = model.generate(prompts, sampling_params=sampling_params)
        responses = [response.outputs[0].text for response in responses]
        # import code; code.interact(local=locals())
        if enable_thinking:
            responses = [response.strip().split('</think>')[1].strip() if len(response.strip().split("</think>"))>1 else "" for response in responses]
        # responses = [response.strip().split('\n')[0] for response in responses]
        # import code; code.interact(local=locals())
        # responses = model.beam_search(prompts, sampling_params)
        # responses = [response.sequences[0].text for response in responses]
    else:
        batch_size = 32
        responses = []
        for idx in tqdm(range(0, len(dataset), batch_size), desc="Generating predictions"):
            right_bound = min(idx + batch_size, len(dataset))
            sources = dataset[idx:right_bound]
            inputs = tokenizer(sources, return_tensors="pt", padding=True)
            inputs.to("cuda:0")
            lang_code = lang2long[tgt_lang]
            print(lang_code)
            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_code), max_length=256
            )
            output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True, model_max_length=256)
            responses.extend(output)
    return responses

def load_model_and_tokenizer(model_path, tensor_parallel_size):
    """Load model and tokenizer."""
    if "nllb" not in model_path:
        model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=4096,
            task="generate",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, token=True, src_lang="eng_Latn"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=True)
        model.to("cuda:0")
    return model, tokenizer

def evaluate_model(enable_thinking, model_path, model, tokenizer, data_dir, lang_pair, max_tokens=512):
    """Evaluate a model on FLORES-101 dataset using vLLM."""
    
    # Load dataset
    src_dataset, tgt_dataset = load_flores_dataset(data_dir, lang_pair)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_tokens,
        temperature=0.0,  # Use greedy decoding
        top_p=1.0,
        top_k=-1,
        seed=0,
    )
    
    # sampling_params = SamplingParams(
    #     n=1,
    #     max_tokens=max_tokens,
    #     temperature=0.1,
    #     top_p=1,
    #     top_k=20,
    # )
    # sampling_params = BeamSearchParams(beam_width=4, max_tokens=max_tokens)
    predictions = predict(enable_thinking, model, tokenizer, src_dataset, sampling_params, lang_pair, model_path)
    
    
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
    
    output = model.predict(inputs, batch_size=8, gpus=1)
    
    scores, mean_score = output.scores, output.system_score
    return {"mean_score": mean_score, "scores": scores}

def has_content(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def evaluate_single_lang_pair(model_path, data_dir, enable_thinking, model, tokenizer, lang_pair, max_tokens, comet22, xcomet, output_file=None):
    """Evaluate a single language pair."""
    def get_spBLEU(hyps, refs):
        if len(hyps) != len(refs):
            return None
        hyps = [hyp.strip() for hyp in hyps]
        refs = [ref.strip() for ref in refs]
        result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200", force=True).score
        return result
    print(f"Evaluating model {model_path} on {lang_pair}...")
    # if has_content(output_file):
    #     print(f"Output file {output_file} already exists and is non-empty. Skipping evaluation.")
    #     return
    
    sources, references, predictions = evaluate_model(
        enable_thinking,
        model_path, 
        model,
        tokenizer,
        data_dir,
        lang_pair, 
        max_tokens
    )
    # temp = [prediction.split("</think>")[1].strip() if len(prediction.split("</think>")) > 1 else "" for prediction in predictions]  
    # import code; code.interact(local=locals())
    metrics = get_spBLEU(predictions, references)
    comet_score = None
    xcomet_score = None
    
    if comet22:
        comet_score = calculate_comet_score(sources, references, predictions)
        print(f"COMET22 Score: {comet_score['mean_score']:.4f}")
    
    if xcomet:
        xcomet_score = calculate_comet_score(
            sources, references, predictions,
            model_path="/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
        )
        print(f"XCOMET Score: {xcomet_score['mean_score']:.4f}")
    
    print("=====================================")
    print(f"Results for {lang_pair}:")
    print(f"spBLEU: {metrics:.4f}")
    if comet_score:
        print(f"COMET Score: {comet_score['mean_score']:.4f}")
    if xcomet_score:
        print(f"XCOMET Score: {xcomet_score['mean_score']:.4f}")
        
    print(f"source: {sources[0]}")
    print(f"prediction: {predictions[0]}")
    print(f"reference: {references[0]}")
    # import code; code.interact(local=locals())
    # Save results if output_file is provided
    if output_file:
        dirname = os.path.dirname(output_file)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        
        with open(output_file, 'w') as f:
            for src, pred, ref in zip(sources, predictions, references):
                f.write(json.dumps({'src': src, 'pred': pred, 'ref': ref}, ensure_ascii=False) + '\n')
            f.write(f"spBLEU: {metrics:.4f}\n")
            if comet_score:
                f.write(f"COMET Score: {comet_score['mean_score']:.4f}\n")
            if xcomet_score:
                f.write(f"XCOMET Score: {xcomet_score['mean_score']:.4f}\n")
    
    return {
        'lang_pair': lang_pair,
        'spBLEU': metrics,
        'comet_score': comet_score['mean_score'] if comet_score else None,
        'xcomet_score': xcomet_score['mean_score'] if xcomet_score else None
    }

def evaluate_multiple_lang_pairs(model_path, data_dir, enable_thinking, model, tokenizer, source_languages, target_languages, max_tokens, comet22, xcomet, output_dir=None):
    """Evaluate multiple language pairs."""
    # Parse language lists
    src_langs = [lang.strip() for lang in source_languages.split(',')]
    tgt_langs = [lang.strip() for lang in target_languages.split(',')]
    # tgt_langs = [lang for lang in list(lang_dict.keys()) if lang != "ary" and lang != "arz"]
    
    # Generate all language pairs
    lang_pairs = []
    for src in src_langs:
        for tgt in tgt_langs:
            # if src != tgt:
            lang_pairs.append(f"{src}-{tgt}")
    
    print(f"Evaluating {len(lang_pairs)} language pairs: {lang_pairs}")

    results = []
    for lang_pair in lang_pairs:
        # Generate output file path if output_dir is provided
        output_file = None
        if output_dir:
            output_file = os.path.join(output_dir, f"{lang_pair}.txt")
        
        result = evaluate_single_lang_pair(
            model_path, data_dir, enable_thinking,
            model, tokenizer,
            lang_pair, max_tokens, 
            comet22, xcomet, output_file
        )
        results.append(result)
    
    # Save summary results
    # if output_dir:
    #     summary_file = os.path.join(output_dir, "summary_results.json")
    #     with open(summary_file, 'w') as f:
    #         json.dump(results, f, indent=2)
    #     print(f"Summary results saved to {summary_file}")
    
    return results

def main():
    parser = transformers.HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.tensor_parallel_size)
    # Determine evaluation mode
    if args.enable_thinking:
        args.max_tokens = 8192
    else:
        args.max_tokens = 512
    if args.lang_pair:
        # Single language pair mode (backward compatibility)
        evaluate_single_lang_pair(
            args.model_name_or_path,
            args.data_dir,
            model,
            tokenizer,
            args.lang_pair,
            args.max_tokens,
            args.comet22,
            args.xcomet,
            args.output_file
        )
    elif args.source_languages and args.target_languages:
        # Multiple language pairs mode
        evaluate_multiple_lang_pairs(
            args.model_name_or_path,
            args.data_dir,
            args.enable_thinking,
            model,
            tokenizer,
            args.source_languages,
            args.target_languages,
            args.max_tokens,
            args.comet22,
            args.xcomet,
            args.output_dir
        )
    else:
        raise ValueError("Either lang_pair or both source_languages and target_languages must be provided")

if __name__ == "__main__":
    main()
