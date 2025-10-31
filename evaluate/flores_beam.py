#!/usr/bin/env python3
"""
FLORES Beam Search Implementation using Hugging Face Transformers

This script implements beam search for machine translation evaluation
on the FLORES dataset using only the Hugging Face library.
"""

import json
import argparse
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig,
    HfArgumentParser
)
from tqdm import tqdm
import sacrebleu
from comet import load_from_checkpoint
import numpy as np
from dataclasses import dataclass, field

# Add the code directory to path for utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from utils import lang_dict


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
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of tokens to generate"}
    )
    num_beams: int = field(
        default=4,
        metadata={"help": "Number of beams for beam search"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for inference"}
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for beam search"}
    )
    no_repeat_ngram_size: int = field(
        default=3,
        metadata={"help": "Size of n-grams to avoid repeating"}
    )
    comet22: bool = field(
        default=False,
        metadata={"help": "Whether to compute COMET22 score"}
    )
    xcomet: bool = field(
        default=False,
        metadata={"help": "Whether to compute XCOMET score"}
    )
    device: str = field(
        default="auto",
        metadata={"help": "Device to use ('auto', 'cuda', 'cpu')"}
    )

def Prompt_template(query, src_language, trg_language):
    instruction = f'Translate the following sentences from {src_language} to {trg_language}.'
    prompt = (
        'Below is an instruction that describes a task, paired with an input that provides further context. '
        'Write a response that appropriately completes the request.\n'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt


class FloresBeamSearch:
    """Beam search implementation for FLORES dataset evaluation."""
    
    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
        comet_model_path: Optional[str] = None
    ):
        """
        Initialize the FLORES beam search evaluator.
        
        Args:
            model_name: Hugging Face model name or path
            tokenizer_name: Tokenizer name (if different from model)
            device: Device to use ('auto', 'cuda', 'cpu')
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            comet_model_path: Path to COMET model for evaluation
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.comet_model_path = comet_model_path
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Load COMET model if provided
        self.comet_model = None
        if self.comet_model_path and os.path.exists(self.comet_model_path):
            print(f"Loading COMET model from {self.comet_model_path}")
            self.comet_model = load_from_checkpoint(self.comet_model_path)
    
    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        print(f"Loading tokenizer: {self.tokenizer_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True,
            padding_side="left",
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load causal LM model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model type: causal")
    
    def generate_with_beam_search(
        self,
        input_texts: List[str],
        lang_pair: str = "eng-eng",
        num_beams: int = 4,
        num_return_sequences: int = 1,
        max_new_tokens: int = 256,
        early_stopping: bool = True,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> List[str]:
        """
        Generate translations using beam search.
        
        Args:
            input_texts: List of source texts to translate
            lang_pair: Language pair in format "src-tgt" (e.g., "eng-hin")
            num_beams: Number of beams for beam search
            num_return_sequences: Number of sequences to return per input
            max_new_tokens: Maximum number of new tokens to generate
            early_stopping: Whether to stop when EOS is generated
            length_penalty: Length penalty for beam search
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            do_sample: Whether to use sampling
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            top_k: Top-k for sampling
            
        Returns:
            List of generated translations
        """
        all_generations = []
        torch.manual_seed(0)
        # Process in batches
        for i in tqdm(range(0, len(input_texts), self.batch_size), desc="Generating"):
            batch_texts = input_texts[i:i + self.batch_size]
            
            # Format prompts based on model type (similar to flores.py)
            formatted_texts = []
            model_path_lower = self.model_name.lower()
            
            # Parse language pair
            src_lang, tgt_lang = lang_pair.split("-")
            src_lang, tgt_lang = lang_dict[src_lang], lang_dict[tgt_lang]
            
            for text in batch_texts:
                if 'llamax' in model_path_lower:
                    # prompt = f"""Translate the following sentences from {src_lang} to {tgt_lang}.\n### Input:\n{text}\n"""
                    prompt = Prompt_template(text.strip(), src_lang, tgt_lang)
                else:
                    prompt = f"{text.strip()}\nTranslate from {src_lang} to {tgt_lang}:\n"
                
                # Apply chat template if available
                try:
                    if 'llamax' in model_path_lower:
                        formatted_prompt = prompt
                    else:
                        message = [{"role": "user", "content": prompt}]
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            message, 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            enable_thinking=False
                        )
                    formatted_texts.append(formatted_prompt)
                except:
                    # Fallback to original prompt if chat template fails
                    formatted_texts.append(prompt)
            # import code; code.interact(local=locals())
            # inputs = self.tokenizer(
            #     formatted_texts,
            #     return_tensors="pt",
            #     padding=True,
            #     truncation=True,
            #     max_length=self.max_length
            # )
            
            # inputs = {k: v.to(self.device) for k, v in inputs.items()}
            tokenized = self.tokenizer(formatted_texts, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.to(self.model.device)
            attn_mask = tokenized.attention_mask.to(self.model.device)
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == self.tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == self.tokenizer.eos_token_id else attn_mask
            # Generation configuration
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                # early_stopping=early_stopping,
                # no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                temperature=temperature,
                # top_p=top_p,
                # top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=input_ids, attention_mask=attn_mask,
                    generation_config=generation_config,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode outputs for causal LM
            # For causal LM, we need to remove the input part
            input_length = input_ids.shape[1]
            generated_ids = outputs[:, input_length:]
            
            # Decode the generated sequences
            batch_generations = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Reshape if we have multiple sequences per input
            if num_return_sequences > 1:
                batch_generations = [
                    batch_generations[i:i + num_return_sequences]
                    for i in range(0, len(batch_generations), num_return_sequences)
                ]
            else:
                batch_generations = [[gen] for gen in batch_generations]
            # import code; code.interact(local=locals())
            all_generations.extend(batch_generations)
        
        # Flatten if single sequence per input
        if num_return_sequences == 1:
            all_generations = [gen[0] for gen in all_generations]
        
        return all_generations
    
    def evaluate_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU score using sacrebleu."""
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        # Clean texts
        predictions = [pred.strip() for pred in predictions]
        references = [ref.strip() for ref in references]
        
        # Calculate BLEU score
        bleu_score = sacrebleu.corpus_bleu(
            predictions, 
            [references], 
            tokenize="flores101", 
            force=True
        ).score
        
        return {"bleu": bleu_score}
    
    def evaluate_comet(self, sources: List[str], predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate COMET score."""
        if not self.comet_model:
            return {"comet": None}
        
        if len(sources) != len(predictions) or len(predictions) != len(references):
            raise ValueError("Number of sources, predictions, and references must match")
        
        # Prepare inputs for COMET
        inputs = [
            {"src": src.strip(), "mt": pred.strip(), "ref": ref.strip()} 
            for src, pred, ref in zip(sources, predictions, references)
        ]
        
        # Calculate COMET score
        output = self.comet_model.predict(inputs, batch_size=16, gpus=1 if self.device == "cuda" else 0)
        
        return {"comet": output.system_score}
    
    def load_flores_data(self, data_path: str) -> Dict[str, List[str]]:
        """Load FLORES dataset from JSONL file."""
        data = {"sources": [], "references": []}
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data["sources"].append(item.get("src", ""))
                    data["references"].append(item.get("ref", ""))
                except json.JSONDecodeError:
                    continue
        
        return data
    
    def my_load_dataset(self, data_pair: str, lang: str) -> List[str]:
        """Load FLORES dataset from directory structure."""
        dataset = []
        path = os.path.join(data_pair, f"{lang}.devtest")
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(line.strip())
        return dataset
    
    def load_flores_dataset(self, data_dir: str, lang_pair: str) -> tuple:
        """Load FLORES-101 dataset for a specific language pair."""
        src_lang, tgt_lang = lang_pair.split("-")
        
        # Get test split
        src_dataset = self.my_load_dataset(data_dir, src_lang)
        tgt_dataset = self.my_load_dataset(data_dir, tgt_lang)
        
        return src_dataset, tgt_dataset
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")


def get_spBLEU(hyps, refs):
    """Calculate spBLEU score."""
    if len(hyps) != len(refs):
        return None
    hyps = [hyp.strip() for hyp in hyps]
    refs = [ref.strip() for ref in refs]
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="spm", force=True).score
    return result

def has_content(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def calculate_comet_score(src_texts, references, predictions, model_path="/mnt/gemini/data1/yifengliu/model/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"):
    """Calculate COMET score."""
    model = load_from_checkpoint(model_path)
    
    # Prepare inputs for COMET
    inputs = [{"src": src.strip(), "mt": mt.strip(), "ref": ref.strip()} for src, mt, ref in zip(src_texts, predictions, references)]
    
    output = model.predict(inputs, batch_size=8, gpus=1)
    
    scores, mean_score = output.scores, output.system_score
    return {"mean_score": mean_score, "scores": scores}


def evaluate_single_lang_pair(model_path, data_dir, evaluator, lang_pair, max_new_tokens, num_beams, batch_size, length_penalty, no_repeat_ngram_size, comet22, xcomet, output_file=None):
    """Evaluate a single language pair."""
    print(f"Evaluating model {model_path} on {lang_pair}...")
    
    # Load dataset
    if has_content(output_file):
        print(f"Output file {output_file} already exists and is non-empty. Skipping evaluation.")
        return
    src_dataset, tgt_dataset = evaluator.load_flores_dataset(data_dir, lang_pair)
    
    # Generate predictions
    predictions = evaluator.generate_with_beam_search(
        input_texts=src_dataset,
        lang_pair=lang_pair,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size
    )
    
    # Calculate metrics
    metrics = get_spBLEU(predictions, tgt_dataset)
    comet_score = None
    xcomet_score = None
    
    if comet22:
        comet_score = calculate_comet_score(src_dataset, tgt_dataset, predictions)
        print(f"COMET22 Score: {comet_score['mean_score']:.4f}")
    
    if xcomet:
        xcomet_score = calculate_comet_score(
            src_dataset, tgt_dataset, predictions,
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
        
    print(f"source: {src_dataset[0]}")
    print(f"prediction: {predictions[0]}")
    print(f"reference: {tgt_dataset[0]}")

    # Save results if output_file is provided
    if output_file:
        dirname = os.path.dirname(output_file)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        print(f"output_file: {output_file}")
        with open(output_file, 'w') as f:
            for src, pred, ref in zip(src_dataset, predictions, tgt_dataset):
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


def evaluate_multiple_lang_pairs(model_path, data_dir, evaluator, source_languages, target_languages, max_new_tokens, num_beams, batch_size, length_penalty, no_repeat_ngram_size, comet22, xcomet, output_dir=None):
    """Evaluate multiple language pairs."""
    # Parse language lists
    src_langs = [lang.strip() for lang in source_languages.split(',')]
    tgt_langs = [lang.strip() for lang in target_languages.split(',')]
    
    # Generate all language pairs
    lang_pairs = []
    for src in src_langs:
        for tgt in tgt_langs:
            lang_pairs.append(f"{src}-{tgt}")
    
    print(f"Evaluating {len(lang_pairs)} language pairs: {lang_pairs}")

    results = []
    for lang_pair in lang_pairs:
        # Generate output file path if output_dir is provided
        output_file = None
        if output_dir:
            output_file = os.path.join(output_dir, f"{lang_pair}.txt")
        
        result = evaluate_single_lang_pair(
            model_path, data_dir, evaluator,
            lang_pair, max_new_tokens, num_beams, batch_size, 
            length_penalty, no_repeat_ngram_size,
            comet22, xcomet, output_file
        )
        results.append(result)
    
    return results


def main():
    """Main function for command line interface."""
    parser = HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Initialize evaluator
    evaluator = FloresBeamSearch(
        model_name=args.model_name_or_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Determine evaluation mode
    if args.lang_pair:
        # Single language pair mode (backward compatibility)
        evaluate_single_lang_pair(
            args.model_name_or_path,
            args.data_dir,
            evaluator,
            args.lang_pair,
            args.max_new_tokens,
            args.num_beams,
            args.batch_size,
            args.length_penalty,
            args.no_repeat_ngram_size,
            args.comet22,
            args.xcomet,
            args.output_file
        )
    elif args.source_languages and args.target_languages:
        # Multiple language pairs mode
        evaluate_multiple_lang_pairs(
            args.model_name_or_path,
            args.data_dir,
            evaluator,
            args.source_languages,
            args.target_languages,
            args.max_new_tokens,
            args.num_beams,
            args.batch_size,
            args.length_penalty,
            args.no_repeat_ngram_size,
            args.comet22,
            args.xcomet,
            args.output_dir
        )
    else:
        raise ValueError("Either lang_pair or both source_languages and target_languages must be provided")


if __name__ == "__main__":
    main()
