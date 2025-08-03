import transformers
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

language_map = {
    'en': 'English',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
}

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    # model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-30B-A3B-Instruct-2507"
    sample = SamplingParams(n=1, temperature=0.6, top_k=-1, top_p=1, max_tokens=32768)
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
    # source_lang, target_lang = "English", "Chinese"
    # template = "Identify if there is any overtranslation in the following English to Chinese translation. Please first explain the reason then give your answer with Yes or No in \\boxed{}. English soure: \"{source_seg}\" Chinese translation: \"{target_seg}\". If there is no overtranslation, answer \"No\". If there is overtranslation, answer \"Yes\" and explain why in detail."
    src_lang = "English"
    tgt_lang = "Swahili"
    source_seg = "Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days."
    # target_seg = "加拿大糖尿病协会临床与科研分会主席、达尔豪斯大学哈利法克斯分校医学院教授埃胡德·厄尔博士指出，目前这项研究仍处于初步阶段，尚需进一步深入探讨。"
    # target_seg = "Dura taa’aan kutaa Kilinikaalaa fi Qorannoo Waldaa Dhukkuba Sukkaaraa Kanaadaa fi Yunivarsiitii Daalhoosii, Haalifaaksiitti Faakulttii Fayyaa keessatti piroofeesara kan ta’an Dr. Ehud Earl qorannoon kun ammallee sadarkaa jalqabaa irra akka jiruu fi marii gadi fageenya qabu dabalataa akka barbaadu ibsaniiru."
    # target_seg = "Dr Ihudii Ur, piroofeesara fayyaa Yuuniiversitii Dalhawuusii Haliifaksii, Noovaa Skoshiyaa fi dura taa'aa kilinikaa fi garee saayinsii Waldaa Dhibee sukkaaraa Kaanadaa kan ta'an ammalle qorannichi guyyoota xiqqoo jalqabaarra akka jiru akeekkachiisaniiru."
    target_seg = "Dk. Ehud Earl, mwenyekiti wa Kitengo cha Kliniki na Utafiti cha Chama cha Kisukari cha Kanada na profesa katika Kitivo cha Tiba katika Chuo Kikuu cha Dalhousie, Halifax, alidokeza kuwa utafiti huu bado uko katika hatua zake za awali na unahitaji mjadala wa kina zaidi."
    # target_seg = "由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。",
    # target_seg = "恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。"
    # source_seg = "Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features.",
    sentence = f"Identify if there is any overtranslation in the following {src_lang} to {tgt_lang} translation. Please first explain the reason then give your answer with Yes or No in \\boxed{{}}. English soure: \"{source_seg}\" Chinese translation: \"{target_seg}\". If there is no overtranslation, answer \"No\". If there is overtranslation, answer \"Yes\" and explain why in detail."
    message = [
        # {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": sentence
        }
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # prompt = f"Translate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}: {sentence}"
    print(f"Prompt: {prompt}")
    output = model.generate(prompt, sample)
    print("==================")
    print(output[0].outputs[0].text)
    import code; code.interact(local=locals())