import logging
from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, MistralForCausalLM
from safetensors.torch import load_file

class RewardModel:
    def __init__(self, model_dir) -> None:
        config = AutoConfig.from_pretrained(model_dir)
        self.device = torch.device('cuda')
        self.model = MistralForCausalLM(config)
        self.model.lm_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        state_dict = load_file(f"{model_dir}/model.safetensors")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(dtype=torch.bfloat16)
        self.model.to(device=self.device)
        self.model.eval()
        logging.info("Load model completed.")

    @torch.no_grad()
    def score(self, prompts, chosens, batch_size: int = 8) -> List[float]:
        # Pre-tokenize all sequences while preserving original structure
        tokenized_seqs = []
        for prompt, chosen in zip(prompts, chosens):
            prompt_tokens = self.tokenizer.encode(prompt)
            chosen_tokens = self.tokenizer.encode(chosen)
            seq = prompt_tokens + chosen_tokens + [self.tokenizer.eos_token_id]
            tokenized_seqs.append(seq)
        
        scores = []
        num_samples = len(tokenized_seqs)
        
        # Process in batches
        for i in range(0, num_samples, batch_size):
            batch_seqs = tokenized_seqs[i:i+batch_size]
            batch_size_actual = len(batch_seqs)
            
            # Create padded tensor with attention mask
            max_len = max(len(seq) for seq in batch_seqs)
            input_ids = torch.full(
                (batch_size_actual, max_len),
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                dtype=torch.long,
                device=self.device
            )
            attention_mask = torch.zeros_like(input_ids)
            
            # Populate tensors and record last token positions
            last_token_positions = []
            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                input_ids[j, :seq_len] = torch.tensor(seq, device=self.device)
                attention_mask[j, :seq_len] = 1
                last_token_positions.append(seq_len - 1)  # Position of last token (our EOS)
            
            # Model forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)  # (batch_size, seq_len)
            
            # Extract scores at last token positions
            batch_scores = logits[torch.arange(batch_size_actual), last_token_positions]
            # import code; code.interact(local=locals())
            scores.extend(batch_scores.tolist())
        
        return scores

if __name__ == '__main__':

    local_model_dir = "/mnt/gemini/data1/yifengliu/model"
    model_dir = f"{local_model_dir}/Seed-X-RM-7B"  
    prompt = ["Translate the following English sentence into Chinese:\nMay the force be with you <zh>", "Translate the following English sentence into Chinese:\nMay the force be with you <zh>"]
    candidate = ["愿原力与你同在","愿力量与你同在"]
    model = RewardModel(model_dir)
    scores = model.score(prompt, candidate, 2)    # output [1.46875, -0.376953125]
    print(scores)
