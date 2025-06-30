import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

language_map = {
    'en': 'English',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
}

if __name__ == '__main__':
    model_path = "/mnt/gemini/data1/yifengliu/checkpoints/Rule-Detect-MetricX-Qwen2.5-0.5B-en-zh-1M-bsz128/global_step780_hf"
    sample = SamplingParams(n=1, temperature=0, top_k=-1, top_p=1, max_tokens=32768)
    src, tgt = "en", "zh"
    model = LLM(model=model_path, max_model_len=32768)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sentence = "If Mr. Trump again wins the presidency, he might order that the federal cases brought by special counsel Jack Smith be dropped by the Justice Department, or even pardon himself to avoid trial.\nDelhi Chief Minister Arvind Kejriwal is the only leader who has defeated Prime Minister Narendra Modi four times in Delhi -- 2013, 2015, 2020 Assembly elections and 2022 MCD polls."
    prompt = f"{sentence}\nTranslate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}:"
    message = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # prompt = f"Translate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}: {sentence}"
    print(f"Prompt: {prompt}")
    output = model.generate(prompt, sample)
    print("==================")
    print(output[0].outputs[0].text)