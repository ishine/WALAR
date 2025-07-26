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
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    model_path = "/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
    sample = SamplingParams(n=1, temperature=0.6, top_k=-1, top_p=1, max_tokens=32768)
    src, tgt = "en", "zh"
    model = LLM(model=model_path, max_model_len=32768)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sentence = """Complete the following sentences in flores101: On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.
Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries.
The JAS 39C Gripen crashed onto a runway at around 9:30 am local time (0230 UTC) and exploded, closing the airport to commercial flights.
The pilot was identified as Squadron Leader Dilokrit Pattavee.
Local media reports an airport fire vehicle rolled over while responding.
28-year-old Vidal had joined Barça three seasons ago, from Sevilla.
Since moving to the Catalan-capital"""
    # prompt = f"{sentence}\nTranslate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}:"
    prompt = sentence
    message = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # prompt = f"Translate from {language_map.get(src, 'English')} to {language_map.get(tgt, 'Chinese')}: {sentence}"
    print(f"Prompt: {prompt}")
    output = model.generate(prompt, sample)
    print("==================")
    print(output[0].outputs[0].text)
    import code; code.interact(local=locals())