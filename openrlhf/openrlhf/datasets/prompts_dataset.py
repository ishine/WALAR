from torch.utils.data import Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")
from utils import write_to_file, preprocess_dataset, my_load_dataset, three2two, two2three

def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
            
        # begin for gemma
        # src_lang = data["two_src_lang"]
        # tgt_lang = data["two_tgt_lang"]
        # text = data[input_key]
        # chat = [{"role": "user", "content": [{"type": "text", "source_lang_code": src_lang, "target_lang_code": tgt_lang, "text": text}]}]
        # end for gemma
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.datasources = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            try:
                # begin gemma
                # src_lang = data['two_src_lang']
                # tgt_lang = data['two_tgt_lang']
                # not_supported_langs = ['ast', 'ceb', 'zho_trad', 'kea', 'kam', 'luo', 'ns', 'umb']
                # if src_lang in not_supported_langs or tgt_lang in not_supported_langs:
                #     continue
                # end gemma
                prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template)
                self.prompts.append(prompt)
                self.labels.append(label)
                self.datasources.append(data.get("datasource", "default"))
            except Exception as e:
                continue
    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.datasources[idx], self.prompts[idx], self.labels[idx]
