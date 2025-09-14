import re
import json
import transformers
import os
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

language_map = {
    'en': 'English',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
}

def load_dataset(path):
    """Load dataset from a JSONL file."""
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def load_flores(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  return lines

def extract_boxed_number(answer):
    # Extract the number from a string in the form of \boxed{number}
    r = re.search(r"\\boxed\{(.*?)\}", answer)
    if r is not None:
        return str(r.group(1))
    return None

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # # model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    # model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    # sample = SamplingParams(n=1, temperature=0.6, top_k=-1, top_p=1, max_tokens=32768)
    # src, tgt = "en", "zh"
    # model = LLM(model=model_path, max_model_len=32768, tensor_parallel_size=1, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    # src = "There was an international drug ring based out of Jamaica at that time who had dealings with South America and it's my understanding the ring would use high-dollar art as collateral in deals."
    # # 当时有一个国际贩毒集团总部设在牙买加，与南美有业务往来，据我所知，这个集团在交易中会将高价艺术品作为 抵押品
    # # tgt = "托尼·莫尔博士在南非夸祖鲁-纳塔尔省发现了这种广泛耐药结核病 (XDR-TB)。"
    # # user_prompt = user_prompt.format(src=src, tgt=tgt)
    # sentence = f"""{src}\nTranslate from English to Chinese:\n"""
    # message = [
    #     {"role": "user", "content": sentence}
    # ]
    # prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # outputs = model.generate([prompt], sample)
    # print(outputs[0].outputs[0].text)
    #     # output = outputs[0].outputs[0].text
    
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/base_en-mix-mid2-1m.jsonl"
    save_path = "/mnt/gemini/data1/yifengliu/qe-lr/data/train/3base_en-mix-mid2-1m.jsonl"
    dataset = load_dataset(save_path)
    # dataset = dataset[220000:]
    # with open(save_path, 'w') as f:
        # for data in dataset:
            # f.write(json.dumps(data, ensure_ascii=False) + "\n")
    import code; code.interact(local=locals())