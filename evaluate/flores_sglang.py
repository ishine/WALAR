import json
import os
import datasets
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
from comet import load_from_checkpoint, download_model
import openai

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
    port: int = field(
        default=1234,
        metadata={"help": "Port for the OpenAI API server"}
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

def predict(model_name_or_path, url, dataset, lang_pair):
    """Generate predictions using the model."""
    src_lang, tgt_lang = lang_pair.split("-")
    lang_dict = {
        'eng': "English",
        "zho_simpl": "Chinese",
        'swh': "Swahili",
        "tam": "Tamil",
        "fra": 'French',
        "rus": "Russian",
        "deu": "German",
    }
    src_lang, tgt_lang = lang_dict[src_lang], lang_dict[tgt_lang]
    client = openai.Client(base_url=url, api_key="None")
    responses = []
    for src_text in tqdm(dataset, desc="Generating predictions"):
        # <X> \n Translate from [SRC] to [TGT]: \n <Y>
        prompt = f"{src_text}\nTranslate from {src_lang} to {tgt_lang}:\n"
        prompt_dict = {'role': 'user', 'content': prompt}
        response = client.chat.completions.create(
            model=model_name_or_path,
            messages=[prompt_dict],
            max_tokens=1024,
            temperature=0,
            n=1,
        )
        responses.append(response.choices[0].message.content)
    return responses


def calculate_metrics(references, predictions):
    """Calculate evaluation metrics."""
    sp_bleu_score = get_spBLEU(predictions, references)
    return sp_bleu_score

def calculate_comet_score(src_texts, references, predictions, model_path="/mnt/gemini/data1/yifengliu/model/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"):
    """Calculate COMET score."""
    # model_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    model = load_from_checkpoint(model_path)
    
    # Prepare inputs for COMET
    inputs = [{"src": src.strip(), "mt": mt.strip(), "ref": ref.strip()} for src, mt, ref in zip(src_texts, predictions, references)]
    
    output = model.predict(inputs)
    
    scores, mean_score = output.scores, output.system_score
    return {"mean_score": mean_score, "scores": scores}

def main():
    parser = transformers.HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    url = f"http://127.0.0.1:{args.port}/v1"
    print(f"Evaluating model {args.model_name_or_path} on {args.lang_pair}...")
    
    sources, references = load_flores_dataset(args.data_dir, args.lang_pair)
    sources, references = sources[:len(sources)], references[:len(references)]
    predictions = predict(args.model_name_or_path, url, sources, args.lang_pair)
    metrics = get_spBLEU(predictions, references)
    comet_score = calculate_comet_score(
        sources, references, predictions
    )
    print("=====================================")
    print(f"args.model_name_or_path: {args.model_name_or_path}")
    print(f"Results for {args.lang_pair}:")
    print(f"spBLEU: {metrics:.4f}")
    print(f"COMET Score: {comet_score['mean_score']:.4f}")
        
    print(f"source: {sources[0]}")
    print(f"prediction: {predictions[0]}")
    print(f"reference: {references[0]}")
    import code; code.interact(local=locals())

    dirname = os.path.dirname(args.output_file) if args.output_file else None
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for src, pred, ref in zip(sources, predictions, references):
                f.write(json.dumps({'src': src, 'pred': pred, 'ref': ref}) + '\n')
            f.write(f"spBLEU: {metrics:.4f}\n")
            f.write(f"COMET Score: {comet_score['mean_score']:.4f}\n")

if __name__ == "__main__":
    main()

