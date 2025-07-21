import json
import random
import pandas as pd
import re
import os
import sacrebleu

from comet import load_from_checkpoint, download_model

def load_dataset(file_path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def get_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    hyps = [hyp.strip() for hyp in hyps]
    refs = [ref.strip() for ref in refs]
    import code; code.interact(local=locals())
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="spm", force=True).score
    return result

def calculate_comet_score(src_texts, references, predictions, model_path="/mnt/gemini/data1/yifengliu/model/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"):
    """Calculate COMET score."""
    model = load_from_checkpoint(model_path)
    
    # Prepare inputs for COMET
    inputs = [{"src": src.strip(), "mt": mt.strip(), "ref": ref.strip()} for src, mt, ref in zip(src_texts, predictions, references)]
    
    output = model.predict(inputs)
    
    scores, mean_score = output.scores, output.system_score
    return {"mean_score": mean_score, "scores": scores}

if __name__ == "__main__":
    # Example usage
    dataset = []
    path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Detect-MetricX-Qwen3-4B-en-zh-1M-bsz128/global_step120_hf/eng-zho_simpl.txt"
    # dataset = load_dataset(path)
    # dataset = load_dataset(path)
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Detect-MetricX-Qwen2.5-3B-Instruct-en-mix-1M-bsz128/global_step180_hf/eng-zho_simpl.txt"
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines[:-3]:
            dataset.append(json.loads(line.strip()))
    # srcs = [data['src'] for data in dataset]
    # hyps = [data['pred'] for data in dataset]
    # refs = [data['ref'] for data in dataset]
    # score = calculate_comet_score(srcs, refs, hyps)
    
    # with open(path, 'a') as f:
    #     f.write(f"COMET Score: {score['mean_score']:.4f}\n")
    # score = get_spBLEU(hyps, refs)
    # tgt_list = ["ara, bel, ben, deu, fin, glg, eng-hin"]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # xcomet_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen2.5-3B-Instruct/eng-tur.txt"
    # with open(path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines[:-2]:
    #         data = json.loads(line.strip())
    #         dataset.append(data)
    # model = load_from_checkpoint(xcomet_path)
    
    # inputs = [{"src": data['src'].strip(), "mt": data['pred'].strip(), "ref": data['ref'].strip()} for data in dataset]
    # output = model.predict(inputs)
    
    # scores, mean_score = output.scores, output.system_score
    
    
    
    
    import code; code.interact(local=locals())