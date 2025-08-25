import json
import random
import pandas as pd
import re
import os
import sacrebleu
import fasttext
from bleurt import score
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
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores101", force=True).score
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
    # dataset = []
    # path = "/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/en/en1m.jsonl"
    # save_path = "/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/en/en1m.jsonl"
    # dataset = load_dataset(path)
    # new_dataset = []
    # for data in dataset:
    #     for dt in data:
    #         dt['label_key'] = dt.pop('ref')
    #         new_dataset.append(dt)
    # with open(save_path, 'w') as f:
    #     for data in new_dataset:
    #         f.write(json.dumps(data) + "\n")
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
        # srcs = [data['src'] for data in dataset]
        # hyps = [data['pred'] for data in dataset]
        # refs = [data['ref'] for data in dataset]
    #     xcomet_path = "/mnt/gemini/data1/yifengliu/model/models--Unbabel--XCOMET-XL/snapshots/6a123c5e8e6dccab25e5fcffa3c8b417abadb462/checkpoints/model.ckpt"
    #     score = calculate_comet_score(srcs, refs, hyps, model_path=xcomet_path)
    #     print(lines[-2])
    #     print(lines[-1])
    #     print(f"{path}: {score['mean_score']}")
    #     with open(path, 'a') as f:
    #         f.write(f"XCOMET Score: {score['mean_s
    # core']:.4f}\n")
    # Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0


    # 26.0056
    dataset = []
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen2.5-3B-Instruct-En-Zh-1M/global_step160_hf/eng-zho_simpl.txt"
    path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Detect-MetricX-Qwen3-4B-en-az-1M-bsz128/global_step260_hf/eng-hye.txt"
    with open(path, 'r') as f:
        lines = f.readlines()
        # import code; code.interact(local=locals())
        for line in lines[:-3]:
            dataset.append(json.loads(line))
    srcs = [data['src'] for data in dataset]
    hyps = [data['pred'] for data in dataset]
    refs = [data['ref'] for data in dataset]
    
    # lang_detect_model = fasttext.load_model("/mnt/gemini/data1/yifengliu/model/lid.176.bin")
    # tgts = ["Danius说，“现在我们正在做 nothing。我已经打电话并发送电子邮件给他的最亲近的合作者，并收到了非常友好的回复。目前来说，这已经足够了。”",
    #         "达尼厄斯说道：“目前我们保持按兵不动。我给他关系最好的合作者打过电话并发送了电子邮件，而且收到了对方非常友好的回复。就目前而言，这足够了。”",
    #         "Weitere Nominierungen gibt es zum Beispiel für den besten Film, die Regie, die Kameraführung, das Kostümbild, den Schnitt, die Original-Filmmusik, das Produktionsdesign, den Tonschnitt, die Tonmischung und das Original-Drehbuch.",
    #         "Die anderen Nominierungen umfassen Best Picture, Director, Cinematography, Costume Design, Film-editing, Original Score, Production Design, Sound Editing, Sound Mixing und Original Screenplay.",
    #         "加拿大糖尿病协会（Canadian Diabetes Association）临床与科学分会主席、达尔豪斯大学（Dalhousie University）哈利法克斯（Halifax, Nova Scotia）医学院教授埃胡德·乌（Ehud Ur）警告说，这项研究仍处于早期阶段。",
    #         "埃胡德·乌尔博士（新斯科舍省哈利法克斯市达尔豪西大学医学教授，加拿大糖尿病协会临床与科学部门教授）提醒，这项研究仍处在早期阶段。",
    #         "周一，瑞典学院诺贝尔文学委员会常务秘书萨拉·丹尼尔斯在瑞典广播电台的一档节目中向公众宣布，委员会因无法直接联系到鲍勃·迪伦，通知他获得了 2016 年诺贝尔文学奖，已经放弃了与他联系的尝试。"]
    # result = lang_detect_model.predict(tgts)
    # model = AutoModelForCausalLM.from_pretrained('/mnt/gemini/data1/yifengliu/model/Qwen3-4B')
    # a = ""
    # b = ""
    # score = get_spBLEU([a], [b])
    
    # dataset[0]['label']=True
    # save_path = "/mnt/gemini/data1/yifengliu/qe-lr/est.jsonl"
    # with open(save_path, 'w') as f:
    #     for data in dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
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