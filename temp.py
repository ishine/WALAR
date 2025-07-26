import json
import random
import pandas as pd
import re
import os
import sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
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
    # path = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Detect-MetricX-Qwen2.5-0.5B-en-zh-1M-bsz128/global_step780_hf/eng-zho_simpl.txt"
    
    # tgt_list = ["zho_simpl", "ara", "deu", "spa", "fin", "jpn", "rus"]
    # for tgt in tgt_list:
    #     path = f"/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Qwen3-32B-AWQ-DA-Qwen2.5-3B-Instruct-en-zh-1M-bsz128/global_step90_hf/eng-{tgt}.txt"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    #     # dataset = load_dataset(path)
    #     # dataset = load_dataset(path)
    #     # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Detect-MetricX-Qwen2.5-3B-Instruct-en-mix-1M-bsz128/global_step180_hf/eng-zho_simpl.txt"
    #     with open(path, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines[:-2]:
    #             dataset.append(json.loads(line.strip()))
    #     srcs = [data['src'] for data in dataset]
    #     hyps = [data['pred'] for data in dataset]
    #     refs = [data['ref'] for data in dataset]
    #     xcomet_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    #     score = calculate_comet_score(srcs, refs, hyps, model_path=xcomet_path)
    #     print(lines[-2])
    #     print(lines[-1])
    #     print(f"{path}: {score['mean_score']}")
    #     with open(path, 'a') as f:
    #         f.write(f"XCOMET Score: {score['mean_score']:.4f}\n")
    # Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0



    # Load the model
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # model = SentenceTransformer("/mnt/gemini/data1/yifengliu/model/Qwen3-Embedding-8B")
    # task_description = "Given the source sentence here"
    # prompt = f"Instruct: Given a source sentence in English, select the most similar corresponding translation in Chinese.\Translation:"
    # # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    # # together with setting `padding_side` to "left":
    # # model = SentenceTransformer(
    # #     "Qwen/Qwen3-Embedding-8B",
    # #     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
    # #     tokenizer_kwargs={"padding_side": "left"},
    # # )

    # # The queries and documents to embed
    # queries = [
    #     "Dr. Tony Moll discovered the Extremely Drug Resistant Tuberculosis (XDR-TB) in the South African region KwaZulu-Natal.",
    # ]
    # documents = [
    #     "Dr. Tony Moll在南非KwaZulu-Natal地区发现了一种非常难治疗的结核病类型——Extremely Drug Resistant Tuberculosis（XDR-TB）。这种病菌对大多数常规抗生素治疗无效，需要使用特定的抗结核药物进行治疗。",
    #     "Dr. Tony Moll在南非KwaZulu-Natal地区发现了一种非常难治疗的结核病类型——Extremely Drug Resistant Tuberculosis（XDR-TB）。"
    # ]

    # # Encode the queries and documents. Note that queries benefit from using a prompt
    # # Here we use the prompt called "query" stored under `model.prompts`, but you can
    # # also pass your own prompt via the `prompt` argument
    # query_embeddings = model.encode(queries, prompt=prompt)
    # document_embeddings = model.encode(documents)

    # # Compute the (cosine) similarity between the query and document embeddings
    # similarity = model.similarity(query_embeddings, document_embeddings)
    # print(similarity)
    
    # model = AutoModelForCausalLM.from_pretrained("/mnt/gemini/data1/yifengliu/model/Qwen3-4B")
    # for name, param in model.named_parameters():
    #     # if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
    #     parts = name.split(".")
    #     print(name, param.shape, param.requires_grad)
    
    
    
    # score = get_spBLEU(hyps, refs)
    # tgt_list = ["ara, bel, ben, deu, fin, glg, eng-hin"]
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