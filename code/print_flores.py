import json
import os
from utils import training_langs

def print_result(file_path):
    bleu_score, xcomet_score = None, None
    metricx_score = None
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'spBLEU' in line: 
                bleu_score = line.strip().split()[-1]
            if 'XComet' in line:
                xcomet_score = line.strip().split()[-1]
            if "MetricX" in line:
                metricx_score = line.strip().split()[-1]
    lang_pair = file_path.split('/')[-1].replace('.txt', '')
    if xcomet_score and metricx_score:
        print(f"{lang_pair}:\t{bleu_score}\t{xcomet_score}\t{metricx_score}")
    elif xcomet_score:
        print(f"{lang_pair}:\t{bleu_score}\t{xcomet_score}")
    else:
        print(f"{lang_pair}:\t{bleu_score}\t")
    return {lang_pair: (bleu_score, xcomet_score, metricx_score)}
    
def check_result(file_path):
    bleu_score, xcomet_score, metricx_score = None, None, None
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'spBLEU' in line: 
                if bleu_score is not None:
                    print(f"spBLEU: {file_path}")
                    return
                bleu_score = line.strip().split()[-1]
            if 'XComet' in line:
                if xcomet_score is not None:
                    print(f"XComet: {file_path}")
                    return
                xcomet_score = line.strip().split()[-1]
            if "MetricX" in line:
                if metricx_score is not None:
                    print(f"MetricX: {file_path}")
                    return
                metricx_score = line.strip().split()[-1]
    return

if __name__ == "__main__":
    dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Mask-Detect-New-Align-Rule-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step400_hf"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/New-Align-Rule-Detect-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step340_hf"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Llama-3.2-3B-Instruct"
    # Walk through the directory and print all file paths
    whole_dict = {}
    src_lang = "eng"
    # tgt_langs_i_care = training_langs
    tgt_langs_i_care = ["ltz", "mkd","pol","srp","slk","slv","ben","guj","hin", "mar", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav", "ara", "tur", "tam", "fin"]
    print(f"Direction\tspBLEU\tXComet\tMetricX")
    for root, dirs, files in os.walk(dir_path):
        for tgt in tgt_langs_i_care:
            file_path = os.path.join(root, f"{src_lang}-{tgt}.txt")
            # file_path = os.path.join(root, file)
            check_result(file_path)
            whole_dict.update(print_result(file_path))
    # print(len(tgt_langs_i_care))
    # import code; code.interact(local=locals())
    # print([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None])
    # print(len([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None]))
    print("Average spBLEU: ", sum([float(whole_dict[f"{src_lang}-{tgt}"][0]) for tgt in tgt_langs_i_care])/len(tgt_langs_i_care))
    # print("Average XComet: ", sum([float(whole_dict[f"{src_lang}-{tgt}"][1]) for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None])/len([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None]))
    # print("Average MetricX: ", sum([float(whole_dict[f"{src_lang}-{tgt}"][2]) for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][2] is not None])/len([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][2] is not None]))
    
    # print the average here
    # import code; code.interact(local=locals())
    # Below is only for copying purpose
    list_of_strings = []
    for tgt in tgt_langs_i_care:
        lang_pair = f"{src_lang}-{tgt}"
        if whole_dict[lang_pair][1] is not None and whole_dict[lang_pair][2] is not None:
            list_of_strings.append(f"{float(whole_dict[lang_pair][0]):.2f} / {float(whole_dict[lang_pair][1])*100:.2f} / {-float(whole_dict[lang_pair][2]):.2f}")
        elif whole_dict[lang_pair][1] is not None:
            list_of_strings.append(f"{float(whole_dict[lang_pair][0]):.2f} / {float(whole_dict[lang_pair][1])*100:.2f}")
        else:
            list_of_strings.append(f"{float(whole_dict[lang_pair][0]):.2f}")
        
        
    temp_file = "output_for_sheets.txt"
    with open(temp_file, "w") as f:
        f.write("\t".join(list_of_strings))
    