from typing import List

import re
import sacrebleu
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch

def get_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="spm", force=True).score
    return result

def make_back_translation_prompts(rollout_samples, tokenizer, model_path):
    all_queries = sum(
        [
            tokenizer.batch_decode(
                remove_pad_token(s.sequences, s.attention_mask), skip_special_tokens=False
            )
            for s in rollout_samples
        ],
        [],
    )
    pattern = r"<\|im_start\|>user\n(.*?)Translate from (.*?) to (.*?):"
    src_langs = [re.search(pattern, q, re.DOTALL).group(2).strip() for q in all_queries]
    tgt_langs = [re.search(pattern, q, re.DOTALL).group(3).strip() for q in all_queries]
    new_prompt = []
    if 'Qwen3' in model_path:
        tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>\n\n(.*?)<\|im_end\|>"
        # tgts = [re.search(tgt_pattern, q, re.DOTALL).group(2).strip() for q in all_queries]
        tgts = [
            match.group(2).strip() if (match := re.search(tgt_pattern, q, re.DOTALL)) else ""
            for q in all_queries
        ]
    else:
        tgt_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
        tgts = []
        tgts = [
            match.group(1).strip() if (match := re.search(tgt_pattern, q, re.DOTALL)) else ""
            for q in all_queries
        ]
        # tgts = [re.search(tgt_pattern, q, re.DOTALL).group(1).strip() for q in all_queries]
    for src_lang, tgt_lang, tgt in zip(src_langs, tgt_langs, tgts):
        sentence = f"{tgt}\nTranslate from {tgt_lang} to {src_lang}:"
        message = [{"role": "user", "content": sentence}]
        new_prompt.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    print(f"new_prompt: {new_prompt[0]}")
    return new_prompt

def calculate_bleu_reward(rollout_samples, back_translate_samples, tokenizer, model_path):
    """Calculate the BLEU reward for the back-translation samples."""
    all_queries = sum(
        [
            tokenizer.batch_decode(
                remove_pad_token(s.sequences, s.attention_mask), skip_special_tokens=False
            )
            for s in rollout_samples
        ],
        [],
    )
    all_back_translations = sum(
        [
            tokenizer.batch_decode(
                remove_pad_token(s.sequences, s.attention_mask), skip_special_tokens=False
            )
            for s in back_translate_samples
        ],
        [],
    )
    pattern = r"<\|im_start\|>user\n(.*?)Translate from (.*?) to (.*?):"
    refs = [re.search(pattern, q, re.DOTALL).group(1).strip() for q in all_queries]
    if 'Qwen3' in model_path:
        # print(all_back_translations)
        hyps = []
        for q in all_back_translations:
            print(q)
            tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>\n\n(.*?)<\|im_end\|>"
            match = re.search(tgt_pattern, q, re.DOTALL)
            if match:
                hyps.append(match.group(2).strip())
            else:
                hyps.append("")
            # if "<|im_end|>" in q:
                # tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>\n\n(.*?)<\|im_end\|>"
            # else:
                # tgt_pattern = r"<\|im_start\|>assistant\n<think>(.*?)</think>\n(.*?)"
            # hyp = re.search(tgt_pattern, q, re.DOTALL).group(2).strip()
            # hyps.append(hyp)
            
        # hyps = [re.search(tgt_pattern, q, re.DOTALL).group(2).strip() for q in all_back_translations]
    else:
        tgt_pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
        hyps = [
            match.group(1).strip() if (match := re.search(tgt_pattern, q, re.DOTALL)) else ""
            for q in all_back_translations
        ]
        hyps = [re.search(tgt_pattern, q, re.DOTALL).group(1).strip() for q in all_back_translations]
    bleu_reward_list = []
    for tgt, label in zip(hyps, refs):
        bleu_score = get_spBLEU([tgt], [label])
        bleu_reward_list.append(bleu_score)
    print(f"In BLEU Ref: {refs[0]}")
    print(f"In BLEU Hyp: {hyps[0]}")
    print(f"BLEU reward: {bleu_reward_list[:10]}")
    return bleu_reward_list

def my_load_dataset(data_pair, lang):
    dataset = []
    path = os.path.join(data_pair, f"{lang}.dev")
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(line.strip())
    return dataset

def load_flores_dataset(data_dir, lang_pair):
    """Load FLORES-101 dataset for a specific language pair."""
    # dataset = load_dataset("facebook/flores", "all")
    src_lang, tgt_lang = lang_pair.split("-")
    
    # Get test split
    src_dataset, tgt_dataset = my_load_dataset(data_dir, src_lang), my_load_dataset(data_dir, tgt_lang)
    
    # Filter for the specific language pair
    return src_dataset, tgt_dataset