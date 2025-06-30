import os
import argparse
from datasets import Dataset, load_dataset
from tqdm import tqdm
import json
import random
import pandas as pd
from transformers import AutoTokenizer

# Language code to language name mapping table
language_map = {
    'en': 'English',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'sw': 'Swahili',
    'ta': 'Tamil',
    "ru": "Russian",
}

def make_prompt(source, src, tgt, template_type='chat', tokenizer=None):
    if template_type == 'base':
        return f"{source}\nTranslate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}:"
    elif template_type == 'chat':
        return f"You are a helpful assistant. Translate this text from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}:\n{source}"
    elif template_type == 'rl':
        return f"Translate this text from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}:\n{source}"
    else:
        raise ValueError(f"Unknown template type: {template_type}")

def read_jsonl_files(file_paths):
    data = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Directly parse as dictionary, no nesting needed
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[Line {line_num}] JSON parse failed → Line content: {repr(line)}")
    return data

def my_load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(json.loads(line))
    return dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare translation dataset')
    parser.add_argument('--src', type=str, default='en', help='Source language code')
    parser.add_argument('--tgt', type=str, default='zh', help='Target language code')
    parser.add_argument('--type', type=str, default="openrlhf", help="Type of dataset to prepare, e.g., openrlhf, verl, etc.")
    parser.add_argument('--input_file', type=str, help='Training JSONL files')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer')
    parser.add_argument('--template_type', type=str, choices=['base', 'chat', 'rl'], default='chat', help='Template type for prompts')
    parser.add_argument('--output_file', type=str, help='Number of training samples to use')
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Read training data
    print(args.input_file)
    data = my_load_dataset(args.input_file)
    dataset = Dataset.from_list(data)

    def make_map_fn(split):
        def process_fn(example, idx):
            data_source = example.get('data_source', 'unknown')
            # Dynamic source and target language field extraction
            source = example['src']
            lg = args.src + "-" + args.tgt
            # Generate prefix
            prompt = make_prompt(source, args.src, args.tgt, template_type=args.template_type)
            
            data = {
                "data_source": data_source + "_" + lg,
                "lang_pair": lg,
                "src_text": source,
                "input_key": prompt,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "translate",
                "extra_info": {
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)
        
    # Save datasets
    dir_name = os.path.dirname(args.output_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if args.type == 'openrlhf':
        train_dataset.to_json(args.output_file, lines=True)
    elif args.type == 'verl':
        train_dataset.to_parquet(args.output_file)

    # Print dataset format
    print("Parquet dataset format:")

    print("Train dataset columns:")
    train_pdf = train_dataset.to_pandas()
    print(train_pdf.head())
    print(train_pdf['prompt'][0])
    
    print(f"Train dataset saved to: {args.output_file}")


if __name__ == '__main__':
    main()


