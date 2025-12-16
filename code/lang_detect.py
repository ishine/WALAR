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
import re
import fasttext
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from utils import lang_dict, mm_dict, long2lang, three2two

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
    use_glotlid: bool = field(
        default=False,
        metadata={"help": "Whether to run detection with the glotLID FastText model outputs"}
    )
    glotlid_model_path: Optional[str] = field(
        default="/mnt/gemini/data1/yifengliu/model/models--cis-lmu--glotlid/snapshots/74cb50b709c9eefe0f790030c6c95c461b4e3b77/model.bin",
        metadata={"help": "Path to the glotLID FastText model (used when --use_glotlid is set)"}
    )
    benchmax_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to BenchMAX style JSON result (e.g., result_en-zh.json)"}
    )
    benchmax_target_language: Optional[str] = field(
        default=None,
        metadata={"help": "Target language code for BenchMAX result (overrides auto-detected code from filename)"}
    )
    benchmax: bool = field(
        default=False,
        metadata={"help": "Enable BenchMAX directory mode (iterate result_<src>-<tgt>.json using lang lists)"}
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


def map_predicted_language(label: str, use_glotlid: bool) -> str:
    """Map a FastText label to a human-readable language name."""
    cleaned_label = label.replace("__label__", "")
    if use_glotlid:
        # glotLID labels are long form like eng_Latn
        return (
            long2lang.get(cleaned_label)
            or lang_dict.get(cleaned_label)
            or mm_dict.get(cleaned_label, cleaned_label)
        )
    return (
        mm_dict.get(cleaned_label)
        or lang_dict.get(cleaned_label)
        or long2lang.get(cleaned_label, cleaned_label)
    )


def convert_to_two_letter(code: str) -> str:
    """Convert ISO code (two or three letters) to a two-letter code as required for BenchMAX filenames."""
    if code is None:
        raise ValueError("Language code cannot be None when converting to two-letter format.")
    stripped = code.strip()
    lowered = stripped.lower()
    if len(lowered) == 2:
        return lowered
    mapped = three2two.get(lowered)
    if mapped:
        return mapped
    raise ValueError(f"Unable to convert language code '{code}' to two-letter format required for BenchMAX.")


def normalize_target_language(code: Optional[str]) -> Optional[str]:
    """Convert a target language code to the English name used by detection outputs."""
    if code is None:
        return None
    stripped = code.strip()
    lowered = stripped.lower()
    return (
        lang_dict.get(stripped)
        or lang_dict.get(lowered)
        or mm_dict.get(stripped)
        or mm_dict.get(lowered)
        or long2lang.get(stripped)
        or long2lang.get(lowered)
        or stripped
    )


def process_file(file_path: str, model, target_language: str, use_glotlid: bool = False) -> float:
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
    target_language = normalize_target_language(target_language)
    for language in lang_info[0]:
        label = language[0] if isinstance(language, (list, tuple)) else language
        pred_lang = map_predicted_language(label, use_glotlid)
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

def detect_single_lang_pair(model, input_dir: str, lang_pair: str, output_file: str = None, use_glotlid: bool = False):
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
    error_rate = process_file(file_path, model, tgt_lang, use_glotlid=use_glotlid)
    
    # Append error rate to file
    append_error_rate_to_file(file_path, error_rate)
    
    print(f"{lang_pair}: {error_rate:.2f}% language error rate")
    
    # Save to output file if specified
    # if output_file:
    #     with open(output_file, 'w') as f:
    #         f.write(f"Lang_Error_Rate: {error_rate:.2f}%\n")


def detect_multiple_lang_pairs(model, input_dir: str, source_languages: str, target_languages: str, output_dir: str = None, use_glotlid: bool = False):
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
            lang_pair, output_file,
            use_glotlid=use_glotlid
        )


def infer_lang_pair_from_filename(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Try to infer (source, target) from filenames like result_en-zh.json."""
    filename = os.path.basename(path)
    match = re.search(r"result_([A-Za-z]+)-([A-Za-z_]+)", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def detect_benchmax_file(model, benchmax_file: str, target_language: str, use_glotlid: bool = False):
    """Process BenchMAX style result JSON and append Lang Consistency score."""
    if not os.path.exists(benchmax_file):
        raise FileNotFoundError(f"BenchMAX file not found: {benchmax_file}")

    with open(benchmax_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    outputs = data.get("outputs", [])
    if not outputs:
        print(f"No outputs found in {benchmax_file}. Skipping language detection.")
        return
    try:
        lang_info = model.predict(outputs)
    except:
        outputs = [output['text'] for output in outputs]
        # import code; code.interact(local=locals())
        lang_info = model.predict(outputs)
    # import code; code.interact(local=locals())
    total_samples = len(outputs)
    target_language = normalize_target_language(target_language)

    mismatch = 0
    for language in lang_info[0]:
        label = language[0] if isinstance(language, (list, tuple)) else language
        pred_lang = map_predicted_language(label, use_glotlid)
        if pred_lang not in target_language:
            mismatch += 1

    consistency = ((total_samples - mismatch) / total_samples) * 100 if total_samples > 0 else 0.0
    data["Lang Consistency"] = round(consistency, 2)

    with open(benchmax_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{benchmax_file}: Lang Consistency = {consistency:.2f}%")


def detect_benchmax_from_dir(
    model,
    benchmax_dir: str,
    source_languages: str,
    target_languages: str,
    use_glotlid: bool = False,
):
    """Iterate BenchMAX directory files using lang lists."""
    if not os.path.isdir(benchmax_dir):
        raise ValueError(f"BenchMAX directory not found: {benchmax_dir}")

    src_langs = [lang.strip() for lang in source_languages.split(',') if lang.strip()]
    tgt_langs = [lang.strip() for lang in target_languages.split(',') if lang.strip()]
    if not src_langs or not tgt_langs:
        raise ValueError("Both source_languages and target_languages must be provided in BenchMAX mode.")

    processed_files = 0
    for src in src_langs:
        src_code = convert_to_two_letter(src)
        for tgt in tgt_langs:
            tgt_code = convert_to_two_letter(tgt)
            file_name = f"result_{src_code}-{tgt_code}.json"
            file_path = os.path.join(benchmax_dir, file_name)
            if not os.path.exists(file_path):
                print(f"[BenchMAX] Skipping missing file: {file_path}")
                continue
            detect_benchmax_file(
                model,
                file_path,
                tgt_code,
                use_glotlid=use_glotlid,
            )
            processed_files += 1

    if processed_files == 0:
        print("[BenchMAX] No files processed. Please verify language codes (two-letter) and directory contents.")


def main():
    import transformers
    parser = transformers.HfArgumentParser(LanguageDetectionArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Load FastText model (glotLID override when requested)
    model_path = args.glotlid_model_path if args.use_glotlid and args.glotlid_model_path else args.model_path
    model = fasttext.load_model(model_path)
    

    if args.benchmax:
        if not args.input_dir:
            raise ValueError("BenchMAX mode requires --input_dir pointing to the BenchMAX output directory.")
        detect_benchmax_from_dir(
            model,
            args.input_dir,
            args.source_languages or "",
            args.target_languages or "",
            use_glotlid=args.use_glotlid,
        )
        return

    # Determine detection mode for legacy flows
    if args.lang_pair:
        # Single language pair mode
        detect_single_lang_pair(
            model,
            args.input_dir,
            args.lang_pair,
            args.output_file,
            use_glotlid=args.use_glotlid
        )
    elif args.source_languages and args.target_languages:
        # Multiple language pairs mode
        detect_multiple_lang_pairs(
            model,
            args.input_dir,
            args.source_languages,
            args.target_languages,
            args.output_dir,
            use_glotlid=args.use_glotlid
        )
    else:
        raise ValueError("Either lang_pair or both source_languages and target_languages must be provided")


if __name__ == "__main__":
    main()

