#!/usr/bin/env python3
"""
Language Detection Module

This module provides language detection functionality using FastText models.
It can process multiple language files and detect the language of text predictions,
calculating the rate of non-target language detection.
"""

import json
import os
import argparse
import fasttext
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from utils import lang_dict, mm_dict

@dataclass
class LanguageDetectionArguments:
    """
    Arguments for language detection.
    """
    model_path: str = field(
        default="/mnt/gemini/data1/yifengliu/model/lid.176.bin",
        metadata={"help": "Path to FastText language detection model"}
    )
    lang_pair: Optional[str] = field(
        default=None,
        metadata={"help": "Language pair for detection (e.g., eng-hin). If provided, overrides source_languages and target_languages."}
    )
    source_languages: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of source languages (e.g., 'eng,deu,fra')"}
    )
    target_languages: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target languages (e.g., 'hin,ben,tam')"}
    )
    input_dir: str = field(
        default="/mnt/gemini/data1/yifengliu/qe-lr/output/flores",
        metadata={"help": "Directory containing input files"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save detection results for all language pairs"}
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save detection results for single language pair"}
    )


def load_dataset(path: str) -> List[str]:
    """Load dataset from a text file (one prediction per line)."""
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # if line and not line.startswith('spBLEU:') and not line.startswith('COMET') and not line.startswith('XCOMET'):
                # dataset.append(line)
            try:
                dataset.append(json.loads(line))
            except:
                break
    return dataset


def detect_language(text: str, model) -> str:
    """Detect language of a single text using FastText model."""
    # Clean the text
    text = text.replace("\n", " ").strip()
    if not text:
        return "unknown"
    
    # Predict language
    lang_info = model.predict([text])
    lang_label = lang_info[0][0].replace("__label__", "")
    return lang_label


def process_file(file_path: str, model, target_language: str) -> float:
    """
    Process a single file and return language detection error rate.
    
    Args:
        file_path: Path to the text file
        model: FastText language detection model
        target_language: Expected target language code
        
    Returns:
        Language detection error rate (percentage)
    """
    if not os.path.exists(file_path):
        return 0.0
    
    dataset = load_dataset(file_path)
    if not dataset:
        return 0.0
    
    # Detect languages
    detected_languages = []
    tgts = [data['pred'] for data in dataset]
    tgts = [tgt.replace("\n", "") for tgt in tgts]
    lang_info = model.predict(tgts)
    cnt = 0
    target_language = lang_dict.get(target_language, target_language)
    for language in lang_info[0]:
        lang_code = language[0].replace("__label__", "")
        pred_lang = mm_dict.get(lang_code, "")
        if pred_lang != target_language:
            cnt += 1
        
        
    # import code; code.interact(local=locals())
    # Calculate error rate
    total_samples = len(dataset)
    error_rate = cnt / total_samples * 100
    # import code; code.interact(local=locals())
    return error_rate


def append_error_rate_to_file(file_path: str, error_rate: float):
    """Append error rate to the end of the file."""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"Lang_Error_Rate: {error_rate:.2f}%\n")

def check_file(file_path):
    """Check if the file already contains a language error rate."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if "Lang_Error_Rate: " in line:
                return True
    return False

def detect_single_lang_pair(model, input_dir: str, lang_pair: str, output_file: str = None):
    """Detect language for a single language pair."""
    src_lang, tgt_lang = lang_pair.split("-")
    
    # Construct file path
    file_path = os.path.join(input_dir, f"{lang_pair}.txt")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    if check_file(file_path):
        print(f"File already contains Lang_Error_Rate: {file_path}")
        return
    # Process file
    error_rate = process_file(file_path, model, tgt_lang)
    
    # Append error rate to file
    append_error_rate_to_file(file_path, error_rate)
    
    print(f"{lang_pair}: {error_rate:.2f}% language error rate")
    
    # Save to output file if specified
    # if output_file:
    #     with open(output_file, 'w') as f:
    #         f.write(f"Lang_Error_Rate: {error_rate:.2f}%\n")


def detect_multiple_lang_pairs(model, input_dir: str, source_languages: str, target_languages: str, output_dir: str = None):
    """Detect language for multiple language pairs."""
    # Parse language lists
    src_langs = [lang.strip() for lang in source_languages.split(',')]
    tgt_langs = [lang.strip() for lang in target_languages.split(',')]
    
    # Generate all language pairs
    lang_pairs = []
    for src in src_langs:
        for tgt in tgt_langs:
            lang_pairs.append(f"{src}-{tgt}")
    
    print(f"Detecting language for {len(lang_pairs)} language pairs...")
    
    for lang_pair in lang_pairs:
        # Generate output file path if output_dir is provided
        output_file = None
        if output_dir:
            output_file = os.path.join(output_dir, f"{lang_pair}_lang_detect.txt")
        
        detect_single_lang_pair(
             model, input_dir,
            lang_pair, output_file
        )


def main():
    import transformers
    parser = transformers.HfArgumentParser(LanguageDetectionArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Load FastText model
    model = fasttext.load_model(args.model_path)
    
    # Determine detection mode
    if args.lang_pair:
        # Single language pair mode
        detect_single_lang_pair(
            model,
            args.input_dir,
            args.lang_pair,
            args.output_file
        )
    elif args.source_languages and args.target_languages:
        # Multiple language pairs mode
        detect_multiple_lang_pairs(
            model,
            args.input_dir,
            args.source_languages,
            args.target_languages,
            args.output_dir
        )
    else:
        raise ValueError("Either lang_pair or both source_languages and target_languages must be provided")


if __name__ == "__main__":
    main()

