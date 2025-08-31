import json
import os

def print_result(file_path):
    bleu_score, xcomet_score = None, None
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'spBLEU' in line: 
                bleu_score = line.strip().split()[-1]
            if 'XComet' in line:
                xcomet_score = line.strip().split()[-1]
    lang_pair = file_path.split('/')[-1].replace('.txt', '')
    if xcomet_score:
        print(f"{lang_pair}:\t{bleu_score}\t{xcomet_score}")
    else:
        print(f"{lang_pair}:\t{bleu_score}\t")
    return {lang_pair: (bleu_score, xcomet_score)}
    

if __name__ == "__main__":
    dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/New-Align-Rule-Detect-MetricX-Qwen3-4B-ar-mix-mid2-1M-bsz128/global_step460_hf"
    # dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Qwen3-4B"
    # Walk through the directory and print all file paths
    whole_dict = {}
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            whole_dict.update(print_result(file_path))
    src_lang = "ara"
    tgt_langs_i_care = ["ltz", "mkd","pol","srp","slk","slv","ben","guj","hin", "mar", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav", "tur", "tam", "fin"]
    # print(len(tgt_langs_i_care))
    print("Average spBLEU: ", sum([float(whole_dict[f"{src_lang}-{tgt}"][0]) for tgt in tgt_langs_i_care])/len(tgt_langs_i_care))
    # print([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None])
    # print(len([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None]))
    print("Average XComet: ", sum([float(whole_dict[f"{src_lang}-{tgt}"][1]) for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None])/len([tgt for tgt in tgt_langs_i_care if whole_dict[f"{src_lang}-{tgt}"][1] is not None]))
    # print the average here
    import code; code.interact(local=locals())
    # Below is only for copying purpose
    list_of_strings = []
    for tgt in tgt_langs_i_care:
        lang_pair = f"{src_lang}-{tgt}"
        if whole_dict[lang_pair][1] is not None:
            list_of_strings.append(f"{float(whole_dict[lang_pair][0]):.2f} / {float(whole_dict[lang_pair][1])*100:.2f}")
        
    temp_file = "output_for_sheets.txt"
    with open(temp_file, "w") as f:
        f.write("\t".join(list_of_strings))
    