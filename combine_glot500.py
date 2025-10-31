#!/usr/bin/env python3
"""
Combine per-language Glot500 JSONL files into a single JSONL.

Reads files like <input_dir>/*/*.jsonl and writes a combined file. Supports
optional per-language limits and shuffling.

Usage:
  python combine_glot500.py --input-dir /path/to/data/glot500 --output /path/to/glot500_combined.jsonl
"""
import os
import json
import glob
import random
import argparse
import sys
from tqdm import tqdm

# allow importing project utils for language mappings
sys.path.append('/mnt/gemini/data1/yifengliu/qe-lr/code')
try:
    from utils import three2two, training_langs2, mm_dict, lang_dict
except Exception:
    three2two = {}
    training_langs2 = []
    mm_dict = {}
    lang_dict = {}

# support list copied from mix_dataset.py; used to filter two-letter codes supported by downstream
support_list = ["af", "als", "am", "an", "ar", "arz", "as", "ast", "av", "az", "azb", "ba", "bar", "bcl", "be", "bg", "bh", "bn", "bo", "bpy", "br", "bs", "bxr", "ca", "cbk", "ce", "ceb", "ckb", "co", "cs", "cv", "cy", "da", "de", "diq", "dsb", "dty", "dv", "el", "eml", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "frr", "fy", "ga", "gd", "gl", "gn", "gom", "gu", "gv", "he", "hi", "hif", "hr", "hsb", "ht", "hu", "hy", "ia", "id", "ie", "ilo", "io", "is", "it", "ja", "jbo", "jv", "ka", "kk", "km", "kn", "ko", "krc", "ku", "kv", "kw", "ky", "la", "lb", "lez", "li", "lmo", "lo", "lrc", "lt", "lv", "mai", "mg", "mhr", "min", "mk", "ml", "mn", "mr", "mrj", "ms", "mt", "mwl", "my", "myv", "mzn", "nah", "nap", "nds", "ne", "new", "nl", "nn", "no", "oc", "or", "os", "pa", "pam", "pfl", "pl", "pms", "pnb", "ps", "pt", "qu", "rm", "ro", "ru", "rue", "sa", "sah", "sc", "scn", "sco", "sd", "sh", "si", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "tyv", "ug", "uk", "ur", "uz", "vec", "vep", "vi", "vls", "vo", "wa", "war", "wuu", "xal", "xmf", "yi", "yo", "yue", "zh"]


def iter_lang_files(input_dir):
    pattern = os.path.join(input_dir, "*", "*.jsonl")
    for p in glob.glob(pattern):
        yield p


def load_jsonl(path, limit=None):
    out = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Try the simple line-by-line read first
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                    continue
                except Exception:
                    # fallthrough to more robust parsing
                    pass

        # If we loaded something already, return (we assume file is proper jsonl)
        if out:
            return out

        # Otherwise try to parse the entire file as multiple JSON objects
        with open(path, 'r', encoding='utf-8') as f:
            whole = f.read().strip()
            if not whole:
                return []
            # Attempt to split by lines containing only '}{' or by newlines between objects
            candidates = []
            # Case: objects concatenated like ...}{... -> insert separator
            if '}{' in whole:
                parts = whole.replace('}{', '}\n{').splitlines()
            else:
                parts = whole.splitlines()

            for i, part in enumerate(parts):
                if limit is not None and i >= limit:
                    break
                s = part.strip()
                if not s:
                    continue
                try:
                    candidates.append(json.loads(s))
                except Exception:
                    # try extracting the first JSON object from the string
                    try:
                        start = s.find('{')
                        end = s.rfind('}')
                        if start != -1 and end != -1 and end > start:
                            obj = json.loads(s[start:end+1])
                            candidates.append(obj)
                    except Exception:
                        continue

            return candidates
    except FileNotFoundError:
        return []


def get_lang(lang_code):
    # map short code to display name using mm_dict/lang_dict; fallback to capitalized code
    src_lang = mm_dict.get(lang_code, '')
    if len(src_lang) == 0:
        src_lang = lang_dict.get(lang_code, '')
    if src_lang == '':
        src_lang = lang_code.capitalize()
    return src_lang


def make_prompt(source, src, tgt, model_name, template_type='chat', tokenizer=None):
    if template_type == 'base':
        if model_name == "llamax":
            # mimic mix_dataset small behavior
            return f"Translate the following sentences from {src} to {tgt}.\n### Input:\n{source}\n"
        else:
            return f"{source}\nTranslate from {src} to {tgt}:"
    elif template_type == 'chat':
        return f"You are a helpful assistant. Translate this text from {src} to {tgt}:\n{source}"
    elif template_type == 'rl':
        return f"Translate this text from {src} to {tgt}:\n{source}"
    else:
        raise ValueError(f"Unknown template type: {template_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['concat', 'mix'], default='concat',
                        help='concat: combine existing jsonl files; mix: build dataset from meta files using mix logic')
    parser.add_argument('--input-dir', type=str, default='/mnt/gemini/data1/yifengliu/data/glot500')
    parser.add_argument('--meta-files', type=str, nargs='*', default=[],
                        help='Explicit meta jsonl files to include when --mode mix')
    parser.add_argument('--output', type=str, default='/mnt/gemini/data1/yifengliu/data/glot500/glot500_combined.jsonl')
    parser.add_argument('--per-lang-limit', type=int, default=None,
                        help='Maximum examples to take from each language file (concat mode)')
    parser.add_argument('--num-per-lang', type=int, default=500,
                        help='Number per target language chunk when in mix mode')
    parser.add_argument('--src-lang-list', type=str, nargs='*', default=["en", "ar", "tr", "hi", "hu", "bg", "id", "es"],
                        help='Source languages for mix mode (two-letter codes)')
    parser.add_argument('--model-name', type=str, default='llamax')
    parser.add_argument('--schedule', action='store_true', help='Follow schedule in mix mode (no global shuffle)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle combined dataset (concat mode or if schedule==False)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    args = parser.parse_args()

    input_dir = args.input_dir
    out_file = args.output

    combined = []
    if args.mode == 'concat':
        files = sorted(list(iter_lang_files(input_dir)))
        if not files:
            print(f"No jsonl files found under {input_dir} (pattern */*.jsonl). Exiting.")
            return
        print(f"Found {len(files)} files. Loading (per-lang-limit={args.per_lang_limit})...")
        for p in files:
            lang = os.path.basename(os.path.dirname(p))
            print(f"  loading {p} (lang={lang})")
            data = load_jsonl(p, limit=args.per_lang_limit)
            # annotate with language if not present
            for d in data:
                if 'data_source' not in d:
                    d['data_source'] = f'glot500_{lang}'
            combined.extend(data)

    else:  # mix mode
        # meta files explicitly provided will be used; additionally append default madlad paths if not provided
        meta_files = list(args.meta_files)
        default_madlad = [
            "/mnt/gemini/data1/yifengliu/data/madlad/bn/clean_docs_v2-00000-of-05000.jsonl",
            "/mnt/gemini/data1/yifengliu/data/madlad/ru/clean_docs_v2-00000-of-05000.jsonl",
            "/mnt/gemini/data1/yifengliu/data/madlad/te/clean_docs_v2-00002-of-05000.jsonl",
            "/mnt/gemini/data1/yifengliu/data/madlad/uz/clean_docs_v2-00001-of-00200.jsonl",
        ]
        for p in default_madlad:
            if p not in meta_files:
                meta_files.append(p)

        num_per_lang = args.num_per_lang
        model_name = args.model_name
        src_lang_list = args.src_lang_list

        # Build lang list from utils.training_langs2 if available, else fallback to empty
        lang_list = list(training_langs2) if training_langs2 else []
        # ensure english included
        if 'eng' not in lang_list:
            lang_list.append('eng')

        all_datasets = []
        for src_lang in src_lang_list:
            # each src_lang has a meta file in meta_files? try to find one matching the folder name
            meta_file_path = None
            for mf in meta_files:
                if f'/{src_lang}/' in mf or mf.split('/')[-1].startswith(src_lang):
                    meta_file_path = mf
                    break
            if meta_file_path is None:
                # fallback: try first meta file
                meta_file_path = meta_files[0] if meta_files else None
            if meta_file_path is None:
                continue

            meta_dataset = load_jsonl(meta_file_path)
            random.shuffle(meta_dataset)

            # construct final_lang_list by filtering to supported two-letter codes
            final_lang_list = []
            for lang in lang_list:
                two_lang = three2two.get(lang, '')
                if two_lang in support_list:
                    final_lang_list.append(lang)

            def make_map_fn(src_lang_name, tgt_lang_name):
                def process_fn(example, idx):
                    data_source = example.get('data_source', 'unknown')
                    source = example.get('src') or example.get('text') or example.get('source') or example.get('text_raw') or ''
                    prompt = make_prompt(source, src_lang_name, tgt_lang_name, model_name, template_type='base')
                    data = {
                        "data_source": data_source + "_" + f"{src_lang_name}-{tgt_lang_name}",
                        "src_lang": src_lang_name,
                        "tgt_lang": tgt_lang_name,
                        "lang_pair": f"{src_lang_name}-{tgt_lang_name}",
                        "src_text": source,
                        "input_key": prompt,
                        "prompt": [{
                            "role": "user",
                            "content": prompt,
                        }],
                        "ability": "translate",
                        "extra_info": {"index": idx}
                    }
                    return data
                return process_fn

            # For each target language chunk, take num_per_lang examples and map
            for i in range(len(final_lang_list)):
                partial = meta_dataset[num_per_lang * i: num_per_lang * (i + 1)]
                tgt_lang = final_lang_list[i]
                tgt_lang_name = get_lang(tgt_lang)
                src_lang_name = get_lang(src_lang)
                for idx, ex in enumerate(partial):
                    mapped = make_map_fn(src_lang_name, tgt_lang_name)(ex, idx)
                    all_datasets.append(mapped)

        # all_datasets already list of dicts
        combined = all_datasets

        # Also include glot500 per-language files into the combined output by default
        include_glot500 = getattr(args, 'include_glot500', False) or (args.mode == 'mix')
        if include_glot500:
            print("Including Glot500 per-language files into final output...")
            glot_files = sorted(list(iter_lang_files(input_dir)))
            for p in glot_files:
                lang = os.path.basename(os.path.dirname(p))
                data_items = load_jsonl(p, limit=args.per_lang_limit)
                for d in data_items:
                    # normalize fields: prefer src_text, preserve data_source
                    unified = {}
                    if 'src' in d:
                        unified['src_text'] = d.get('src')
                    elif 'text' in d:
                        unified['src_text'] = d.get('text')
                    else:
                        # try other fields
                        unified['src_text'] = d.get('src_text') or d.get('source') or ''
                    unified['data_source'] = d.get('data_source', f'glot500_{lang}')
                    if 'length' in d:
                        unified['length'] = d.get('length')
                    combined.append(unified)

    print(f"Loaded total {len(combined)} examples.")
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(combined)

    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_file, 'w', encoding='utf-8') as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"Wrote combined dataset to: {out_file}")


if __name__ == '__main__':
    main()
