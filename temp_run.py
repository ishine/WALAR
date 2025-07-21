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
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    sample = SamplingParams(n=1, temperature=0.6, top_k=-1, top_p=1, max_tokens=32768)
    src, tgt = "en", "zh"
    model = LLM(model=model_path, max_model_len=32768)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sentence = "1 + 1 = ? /no_think"
    # prompt = f"{sentence}\nTranslate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}:"
    prompt = sentence
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
    import code; code.interact(local=locals())