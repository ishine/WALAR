import json
import os
import csv
from utils import training_langs, training_langs2, llamax_langs, high_langs, flores_langs, qwen_langs

def load_dataset(path):
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            dataset.append(data)
    return dataset

if __name__ == "__main__":
    dir_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/flores_devtest/metricX-xxl-bf16"

    whole_dict = {}
    xcomet_not_support_list = ["ltz", "ceb", "yor"]
    metricx_not_support_list = ["ast", "oci", "bos", "hrv", "asm", "ory", "lug", "kea", "kam", "lin", "nya", "wol", "ful", "orm", "luo"]
    
    # Support multiple source languages
    # src_langs_i_care = ["ara", "eng", "fra", "deu", "spa"]  # Add more source languages as needed
    src_langs_i_care = ['eng', 'deu', 'fra', 'spa', 'ita', 'por', 'rus', 'nld', 'bul', 'ind', 'ron', 'mkd', 'hin', 'ces', 'zho_simpl', 'fin', 'hun', 'pol', 'tur', 'ukr', 'ben', 'ara', 'isl']
    tgt_langs_i_care = flores_langs 
    tgt_langs_i_care = [tgt for tgt in tgt_langs_i_care if tgt not in metricx_not_support_list]
    # tgt_langs_i_care = [tgt for tgt in tgt_langs_i_care if tgt not in metricx_not_support_list]

    # Create a dictionary to store predictions: {src_lang: {tgt_lang: prediction}}
    predictions_dict = {}
    
    print(f"Direction\tPred")
    prediction_list = []
    
    for root, dirs, files in os.walk(dir_path):
        for src in src_langs_i_care:
            if src not in predictions_dict:
                predictions_dict[src] = {}
            
            for tgt in tgt_langs_i_care:
                if src != tgt:
                    file_path = os.path.join(root, f"{src}-{tgt}.jsonl")
                    if os.path.exists(file_path):
                        dataset = load_dataset(file_path)
                        predictions = [data['prediction'] for data in dataset]
                        prediction = sum(predictions) / len(predictions)
                        predictions_dict[src][tgt] = prediction
                        print(f"{src}-{tgt}:\t{prediction}")
                        prediction_list.append(prediction)
                    else:
                        print(f"Warning: File {file_path} not found")
                        predictions_dict[src][tgt] = None
    
    print(f"Avg Pred:\t{sum(prediction_list)/len(prediction_list)}")
    
    # Write to CSV file
    output_csv = "/mnt/gemini/data1/yifengliu/qe-lr/qe_predictions.csv"
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        # Get all target languages that appear in the data
        all_tgt_langs = set()
        for src_data in predictions_dict.values():
            all_tgt_langs.update(src_data.keys())
        all_tgt_langs = sorted(list(all_tgt_langs))
        
        # Write header
        writer = csv.writer(csvfile)
        writer.writerow(['Source Language'] + all_tgt_langs + ['Avg'])
        
        # Write data rows
        for src_lang in sorted(predictions_dict.keys()):
            row = [src_lang]
            valid_predictions = []  # Store valid predictions for calculating average
            
            for tgt_lang in all_tgt_langs:
                if tgt_lang in predictions_dict[src_lang] and predictions_dict[src_lang][tgt_lang] is not None:
                    row.append(predictions_dict[src_lang][tgt_lang])
                    valid_predictions.append(predictions_dict[src_lang][tgt_lang])
                else:
                    row.append('')
            
            # Calculate average for this source language
            if valid_predictions:
                avg_prediction = sum(valid_predictions) / len(valid_predictions)
                row.append(avg_prediction)
            else:
                row.append('')
            
            writer.writerow(row)
    
    print(f"Results saved to: {output_csv}")
