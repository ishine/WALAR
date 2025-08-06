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
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    # model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-30B-A3B-Instruct-2507"
    sample = SamplingParams(n=4, temperature=0.6, top_k=-1, top_p=1, max_tokens=32768)
    src, tgt = "en", "zh"
    model = LLM(model=model_path, max_model_len=32768, tensor_parallel_size=2, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
#     system_prompt = "You are a professional translation evaluator."
#     user_prompt = """Your task is to assess whether a translation segment successfully conveys the semantic content of the original sentence according to the following criteria:
# Key Information Recognition: Identify whether the key information in the source (e.g., proper nouns, keywords, terminologies, or sentence structures) is present in the translation.


# Correctness Assessment: Determine whether the translation accurately conveys the source sentence's intention, without misinterpretation or contextual errors.


# Expressiveness Assessment: Evaluate whether the translation is fluent, clear, and intuitive to human readers. It should avoid unnecessary verbosity, ambiguous phrases, or awkward grammar.
# Given a source sentence and its translation, please first analyze the translation and finally answer "Yes" if the translation meets all three criteria and answer "No" otherwise.

# Source sentence: {src}
# Translation: {tgt}
# """
    # src = "There was an international drug ring based out of Jamaica at that time who had dealings with South America and it's my understanding the ring would use high-dollar art as collateral in deals."
    # 当时有一个国际贩毒集团总部设在牙买加，与南美有业务往来，据我所知，这个集团在交易中会将高价艺术品作为 抵押品
    # tgt = "托尼·莫尔博士在南非夸祖鲁-纳塔尔省发现了这种广泛耐药结核病 (XDR-TB)。"
    # user_prompt = user_prompt.format(src=src, tgt=tgt)
    # sentence = f"""{src}\nTranslate from English to Chinese:\n"""
    # template = "Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100, where a score of zero means \"no meaning preserved\" and score of one hundred means \"perfect meaning and grammar\". Please first give your explanation, and finally give your score in \\boxed{{}}.\n\n{source_lang} source: \"{source_seg}\"\n{target_lang} translation: \"{target_seg}\"\n"
    # source_seg = "Danius said, \"Right now we are doing nothing. I have called and sent emails to his closest collaborator and received very friendly replies. For now, that is certainly enough.\""
    # target_seg = "丹尼斯说：“目前我们暂时不采取行动。我已经联系并发邮件给他的主要合作者，对方回复非常友好。目前来说，这已经足够。”"
    source_lang, target_lang = "English", "Chinese"
    src_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/eng.devtest"
    tgt_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/zho_simpl.devtest"
    src_dataset, tgt_dataset = load_flores(src_path), load_flores(tgt_path)
    # src = "Workplace harmony is crucial, emphasizing group effort rather than praising individual accomplishments."
    # template = f"{src}\nTranslate from English to Chinese:\n"
    prompts = []
    for src, tgt in zip(src_dataset, tgt_dataset):
        source_seg = src.strip()
        target_seg = tgt.strip()
        template = f"Identify if there is any overtranslation in the following {source_lang} to {target_lang} translation. Please first explain the reason then give your answer with Yes or No in \\boxed{{}}. English soure: \"{source_seg}\" Chinese translation: \"{target_seg}\". If there is no overtranslation, answer \"No\". If there is overtranslation, answer \"Yes\" and explain why in detail."
        message = [
            {"role": "user", "content": template},
        ]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(prompt)
    outputs = model.generate(prompts, sampling_params=sample)
    outputs1 = [[opt.text for opt in output.outputs] for output in outputs]
    answers = [
        [match if (match := extract_boxed_number(opt)) else "No" for opt in output]
        for output in outputs1
    ]
    final_answers = [Counter(answer).most_common(1)[0][0] for answer in answers]
    correct = sum([answer == "Yes" for answer in final_answers])
    wrong = sum([answer == "No" for answer in final_answers])
        # output = outputs[0].outputs[0].text
    
    # path = "/mnt/gemini/data1/yifengliu/qe-lr/simple_test.jsonl"
    # dataset = load_dataset(path)
    # # num = sum([1 if data.get("label", "") != "" else 0 for data in dataset])
    # correct_oracle = sum([1 if data.get("label", "") == True else 0 for data in dataset])
    # false_oracle = sum([1 if data.get("label", "") == False else 0 for data in dataset])
    # # print(num)
    # src_lang = "English"
    # tgt_lang = "Chinese"
    # prompts = []
    # for data in dataset:
    #     source_seg, target_seg = data['src'], data['pred']
    # # source_seg = "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days."
    # # target_seg = "Dk. Ehud Earl, mwenyekiti wa Kitengo cha Kliniki na Utafiti cha Chama cha Kisukari cha Kanada na profesa katika Kitivo cha Tiba katika Chuo Kikuu cha Dalhousie, Halifax, alidokeza kuwa utafiti huu bado uko katika hatua zake za awali na unahitaji mjadala wa kina zaidi."
    #     sentence = f"Identify if there is any overtranslation in the following {src_lang} to {tgt_lang} translation. Please first explain the reason then give your answer with Yes or No in \\boxed{{}}. English soure: \"{source_seg}\" Chinese translation: \"{target_seg}\"."
    #     message = [
    #         # {"role": "system", "content": system_prompt},
    #         {
    #             "role": "user",
    #             "content": sentence
    #         }
    #     ]
    #     prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    #     # prompt = f"Translate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}: {sentence}"
    #     # print(f"Prompt: {prompt}")
    #     prompts.append(prompt)
    # outputs = model.generate(prompts, sample)
    # outputs1 = [[opt.text for opt in output.outputs] for output in outputs]
    # answers = [
    #     [match if (match := extract_boxed_number(opt)) else "No" for opt in output]
    #     for output in outputs1
    # ]
    # final_answers = [Counter(answer).most_common(1)[0][0] for answer in answers]
    # labels = [data['label'] for data in dataset]
    # tp, fp, tn, fn = 0, 0, 0, 0
    # for label, final_answer in zip(labels, final_answers):
    #     if label == True and final_answer == "Yes":
    #         tp += 1
    #     elif label == False and final_answer == "Yes":
    #         fp += 1
    #     elif label == False and final_answer == "No":
    #         tn += 1
    #     elif label == True and final_answer == "No":
            # fn += 1
    # print("==================")
    # print(output[0].outputs[0].text)
    # print(f"TP: {tp/len(labels)}, FP: {fp/len(labels)}, TN: {tn/len(labels)}, FN: {fn/len(labels)}")
    import code; code.interact(local=locals())