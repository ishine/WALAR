import json
import sentencepiece as spm
from collections import defaultdict

import re
from typing import List, Set, Tuple
from collections import defaultdict
from tqdm import *


tokenizer_path = "/mnt/gemini/data1/yifengliu/model/models--facebook--xlm-roberta-xl/snapshots/aa5d120255845efeebc9b7f42822a1dd0f9ece9d/sentencepiece.bpe.model"
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)

def tokenize_subword(text: str) -> List[str]:
    """Tokenize text into sub-word tokens without normalization."""
    # Using a simple regex-based subword tokenizer
    # This is a simplified version; in practice, use a proper subword tokenizer like WordPiece or BPE
    # tokens = re.findall(r'\w{1,3}|\S', text.lower())
    tokens = sp.encode(text, out_type=str)
    return tokens



# Function to check contamination for a single evaluation example
def check_contamination(eval_example, eight_gram_dict, corpus_tokens):
    # eval_tokens = eval_example.split()  # Tokenize evaluation example
    eval_tokens = eval_example
    n = len(eval_tokens)
    
    # Handle cases with fewer than 8 tokens
    if n < 8:
        return "not contaminated"
    
    max_match_length = 0
    
    # Generate all overlapping 8-grams from the evaluation example
    for i in range(n - 7):
        eight_gram = tuple(eval_tokens[i:i + 8])
        
        # Search for the 8-gram in the corpus
        if eight_gram in eight_gram_dict:
            for j in eight_gram_dict[eight_gram]:
                # Extend the match to the left
                k = 0
                while (i - k - 1 >= 0 and 
                       j - k - 1 >= 0 and 
                       eval_tokens[i - k - 1] == corpus_tokens[j - k - 1]):
                    k += 1
                
                # Extend the match to the right
                l = 0
                while (i + 8 + l < n and 
                       j + 8 + l < corpus_length and 
                       eval_tokens[i + 8 + l] == corpus_tokens[j + 8 + l]):
                    l += 1
                
                # Calculate the length of the extended match
                match_length = 8 + k + l
                if match_length > max_match_length:
                    max_match_length = match_length
    
    # Calculate the overlap percentage
    overlap_percentage = max_match_length / n
    
    # Determine if the example is contaminated (threshold = 0.7)
    if overlap_percentage >= 0.7:
        print("Here!")
        # import code; code.interact(local=locals())
        return True
    else:
        return False



def load_dataset(path):
    """Load a set from a text file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f]

def load_jsonl_dataset(path):
    """Load a JSONL file into a list of dictionaries."""
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

if __name__ == '__main__':
    dev_path = "/mnt/gemini/data1/yifengliu/data/flores101_dataset/dev/eng.dev"
    devtest_path = "/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/eng.devtest"
    pretrain_path = "/mnt/gemini/data1/yifengliu/data/wmt24_news_crawl/en/en1m.jsonl"
    dev_dataset, devtest_dataset = load_dataset(dev_path), load_dataset(devtest_path)
    dev_dataset.extend(devtest_dataset)
    total_set = dev_dataset
    pretrained_dataset = load_jsonl_dataset(pretrain_path)
    pretrained_dataset = [data['src'] for data in pretrained_dataset]
    
    test_data = [sp.encode(text, out_type=str) for text in total_set]
    pretrain_data = [sp.encode(data, out_type=str) for data in pretrained_dataset]
    # eight_gram_dict_list = []
    # for i in tqdm(range(len(pretrain_data)), desc="Building 8-gram dictionaries"):
    #     data = pretrain_data[i]
    #     eight_gram_dict = {}
    #     for j in range(len(data) - 7):
    #         eight_gram = tuple(data[j:j + 8])
    #         if eight_gram not in eight_gram_dict:
    #             eight_gram_dict[eight_gram] = []
    #         eight_gram_dict[eight_gram].append(j)
    #     eight_gram_dict_list.append(eight_gram_dict)
    
    corpus_tokens = [dt for data in pretrain_data for dt in data]
    corpus_length = len(corpus_tokens)
    eight_gram_dict = {}
    for j in tqdm(range(corpus_length - 7), desc="Building 8-gram dictionary"):
        eight_gram = tuple(corpus_tokens[j:j + 8])
        if eight_gram not in eight_gram_dict:
            eight_gram_dict[eight_gram] = []
        eight_gram_dict[eight_gram].append(j)
    
    total_idx_list = []
    for idx in tqdm(range(len(test_data)), desc="Checking contamination"):
        dt = test_data[idx]
        result = check_contamination(dt, eight_gram_dict, corpus_tokens)
        if result:
            total_idx_list.append(idx)
        # if result:
            # break
            # print("Here!")
            # import code; code.interact(local=locals())
    # results = check_contamination(total_set, pretrained_dataset, threshold=0.7)
    
    import code; code.interact(local=locals())