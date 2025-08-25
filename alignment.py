import json
import torch
import transformers
import itertools
import jieba
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tqdm import *
from collections import defaultdict
from simalign import SentenceAligner
from FlagEmbedding import BGEM3FlagModel

def load_flores(path):
  with open(path, 'r') as f:
    lines = f.readlines()
  return lines

def my_load_dataset(path):
  dataset = []
  with open(path, 'r') as f:
    lines = f.readlines()
    for line in lines[:-3]:
      dataset.append(json.loads(line.strip()))
  return dataset

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def print_alignments(src, tgt, alignments):
  for alignment_pair in sorted(alignments):
    src_idx, trg_idx = alignment_pair
    print(f'{color.BOLD}{color.BLUE}{src[src_idx]}{color.END}==={color.BOLD}{color.RED}{tgt[trg_idx]}{color.END}')

def align_score(srcs, tgts, model, tokenizer):
  def print_alignments(src, tgt, alignments):
    for alignment_pair in sorted(alignments):
      src_idx, trg_idx = alignment_pair
      print(f'{color.BOLD}{color.BLUE}{src[src_idx]}{color.END} === {color.BOLD}{color.RED}{tgt[trg_idx]}{color.END}')
  align_score_list = []
  # for src, tgt in zip(srcs, tgts):
  f1_list = []
  for idx in tqdm(range(len(srcs))):
    sent_src, sent_tgt = srcs[idx], tgts[idx]
    # sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    # print(token_src)
    # print(token_tgt)
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
      sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
      sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 24
    threshold = 1e-3
    model.eval()
    # import code; code.interact(local=locals())
    with torch.no_grad():
      out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
      out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
      
      dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

      softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
      softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

      softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
      align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
    # align_percent = len(align_words) / len(token_tgt)
    src_words = set({t[0]: t for t in sorted(align_words)}.values())
    tgt_words = set({t[1]: t for t in sorted(align_words)}.values())
    precision = min(len(tgt_words) / len(token_tgt), 1)
    recall = min(len(src_words) / len(token_src), 1)
    f1 = 2 * precision * recall / (precision + recall)
    # f1 = [2*precision*recall / (precision + recall) for precision, recall in zip(precisions, recalls)]
    # align_score_list.append(align_percent if align_percent <= 1 else 1)
    # if idx == 1008:
    # print_alignments(sent_src, sent_tgt, src_words)
    # print_alignments(sent_tgt, sent_src, {(b, a) for (a, b) in tgt_words})
    print_alignments(sent_src, sent_tgt, align_words)
    # import code; code.interact(local=locals())
    f1_list.append(f1)
  return f1_list
  # return align_score_list

import torch
import itertools
from tqdm import tqdm

import torch
from tqdm import tqdm

def align_score2(srcs, tgts, model, tokenizer, batch_size=16):
    align_score_list = []
    align_layer = 24
    threshold = 1e-3

    # Precompute all tokenizations and mappings
    tokenized_data = []
        
    for i in tqdm(range(len(srcs))):
      sent_src, sent_tgt = srcs[i], tgts[i]
      # sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
      token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
      # print(token_src)
      # print(token_tgt)
      wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
      ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
      sub2word_map_src = []
      for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
      sub2word_map_tgt = []
      for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]
      tokenized_data.append({
        'input_ids_src': ids_src.squeeze().tolist(),
        'input_ids_tgt': ids_tgt.squeeze().tolist(),
        'sub2word_src': sub2word_map_src,
        'sub2word_tgt': sub2word_map_tgt,
        'src_len': len(sent_src),
        'tgt_len': len(sent_tgt)
      })
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_data), batch_size)):
            batch = tokenized_data[i:i+batch_size]
            
            # Prepare batch inputs
            src_batch = [item['input_ids_src'] for item in batch]
            tgt_batch = [item['input_ids_tgt'] for item in batch]
            
            # Pad batches
            src_tensors = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in src_batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
            tgt_tensors = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in tgt_batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
            src_mask = (src_tensors != tokenizer.pad_token_id).long()
            tgt_mask = (tgt_tensors != tokenizer.pad_token_id).long()
            # import code; code.interact(local=locals())
            # Get model outputs
            out_src = model(src_tensors, attention_mask=src_mask, output_hidden_states=True)[2][align_layer]
            out_tgt = model(tgt_tensors, attention_mask=tgt_mask, output_hidden_states=True)[2][align_layer]
            # import code; code.interact(local=locals())
            # Process each sentence in the batch
            for j, item in enumerate(batch):
                # Remove padding and special tokens
                src_len = sum([1 for x in src_batch[j] if x != tokenizer.pad_token_id])
                tgt_len = sum([1 for x in tgt_batch[j] if x != tokenizer.pad_token_id])
                valid_src = out_src[j, 1:src_len-1]  # Remove [CLS] and [SEP]
                valid_tgt = out_tgt[j, 1:tgt_len-1]
                
                # Calculate alignment
                dot_prod = torch.matmul(valid_src, valid_tgt.transpose(-1, -2))
                softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
                softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)
                softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)
                
                # Convert to word alignment
                align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
                align_words = set()
                for i_sub, j_sub in align_subwords:
                    align_words.add((
                        item['sub2word_src'][i_sub],
                        item['sub2word_tgt'][j_sub]
                    ))
                
                # Calculate scores
                src_words = {t[0] for t in align_words}
                tgt_words = {t[1] for t in align_words}
                n_src = len(src_words)
                n_tgt = len(tgt_words)
                
                precision = n_tgt / item['tgt_len'] if n_tgt > 0 else 0
                recall = n_src / item['src_len'] if n_src > 0 else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                # import code; code.interact(local=locals())
                align_score_list.append(f1)
    
    return align_score_list

def align_score3(srcs, tgts, model, tokenizer, batch_size=16):
    align_score_list = []
    align_layer = 24
    threshold = 1e-3

    # 预计算所有标记化和映射
    tokenized_data = []
    start = time.time()
    for sent_src, sent_tgt in zip(srcs, tgts):
        # 标记化句子并创建子词映射
        enc_src = tokenizer(sent_src, is_split_into_words=True, truncation=True, 
                           return_offsets_mapping=True, return_tensors=None)
        enc_tgt = tokenizer(sent_tgt, is_split_into_words=True, truncation=True, 
                           return_offsets_mapping=True, return_tensors=None)
        
        # 创建子词到词的映射
        sub2word_src = []
        for i, (start, end) in enumerate(enc_src['offset_mapping']):
            if start != end:  # 跳过特殊标记
                sub2word_src.append(enc_src.word_ids()[i])
        
        sub2word_tgt = []
        for i, (start, end) in enumerate(enc_tgt['offset_mapping']):
            if start != end:
                sub2word_tgt.append(enc_tgt.word_ids()[i])
        
        tokenized_data.append({
            'input_ids_src': enc_src['input_ids'],
            'input_ids_tgt': enc_tgt['input_ids'],
            'sub2word_src': sub2word_src,
            'sub2word_tgt': sub2word_tgt,
            'src_len': len(sent_src),
            'tgt_len': len(sent_tgt),
            'src_token_len': len(sub2word_src),  # 有效标记长度
            'tgt_token_len': len(sub2word_tgt)
        })
    mid = time.time()
    print(f"Tokenization time: {mid - start:.2f} seconds")
    # 批量处理
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_data), batch_size)):
            batch = tokenized_data[i:i+batch_size]
            batch_size_actual = len(batch)
            
            # 准备批量输入
            src_batch = [item['input_ids_src'] for item in batch]
            tgt_batch = [item['input_ids_tgt'] for item in batch]
            
            # 填充批次
            src_tensors = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in src_batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
            tgt_tensors = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in tgt_batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
            
            src_mask = (src_tensors != tokenizer.pad_token_id).long()
            tgt_mask = (tgt_tensors != tokenizer.pad_token_id).long()
            # import code; code.interact(local=locals())
            # Get model outputs
            out_src = model(src_tensors, attention_mask=src_mask, output_hidden_states=True)[2][align_layer]
            out_tgt = model(tgt_tensors, attention_mask=tgt_mask, output_hidden_states=True)[2][align_layer]
            
            # 提取有效标记（移除[CLS]和[SEP]）
            valid_src_list = []
            valid_tgt_list = []
            src_token_lengths = []
            tgt_token_lengths = []
            
            for j, item in enumerate(batch):
                # 获取有效标记
                valid_src = out_src[j, 1:1+item['src_token_len']]  # 移除[CLS]和[SEP]
                valid_tgt = out_tgt[j, 1:1+item['tgt_token_len']]
                
                valid_src_list.append(valid_src)
                valid_tgt_list.append(valid_tgt)
                src_token_lengths.append(item['src_token_len'])
                tgt_token_lengths.append(item['tgt_token_len'])
            
            # 找到最大长度以进行填充
            max_src_len = max(src_token_lengths)
            max_tgt_len = max(tgt_token_lengths)
            
            # 填充所有向量以创建批量矩阵
            padded_src = torch.zeros(batch_size_actual, max_src_len, out_src.size(-1), 
                                   device=out_src.device)
            padded_tgt = torch.zeros(batch_size_actual, max_tgt_len, out_tgt.size(-1), 
                                   device=out_tgt.device)
            
            # 创建注意力掩码
            src_mask = torch.zeros(batch_size_actual, max_src_len, device=out_src.device)
            tgt_mask = torch.zeros(batch_size_actual, max_tgt_len, device=out_tgt.device)
            
            for j in range(batch_size_actual):
                src_len = src_token_lengths[j]
                tgt_len = tgt_token_lengths[j]
                
                padded_src[j, :src_len] = valid_src_list[j]
                padded_tgt[j, :tgt_len] = valid_tgt_list[j]
                
                src_mask[j, :src_len] = 1
                tgt_mask[j, :tgt_len] = 1
            
            # 批量计算点积 - 使用爱因斯坦求和约定进行高效矩阵乘法
            # 结果形状: (batch_size, max_src_len, max_tgt_len)
            dot_prod = torch.einsum('bse,bte->bst', padded_src, padded_tgt)
            
            # 批量计算softmax
            # 对源到目标方向应用softmax
            softmax_srctgt = torch.nn.functional.softmax(dot_prod, dim=-1)
            # 对目标到源方向应用softmax
            softmax_tgtsrc = torch.nn.functional.softmax(dot_prod, dim=-2)
            
            # 应用阈值
            softmax_inter = (softmax_srctgt > threshold) & (softmax_tgtsrc > threshold)
            
            # 处理每个句子的对齐
            for j, item in enumerate(batch):
                src_len = src_token_lengths[j]
                tgt_len = tgt_token_lengths[j]
                
                # 获取当前句子的对齐矩阵
                sentence_align = softmax_inter[j, :src_len, :tgt_len]
                
                # 转换为词对齐
                align_subwords = torch.nonzero(sentence_align, as_tuple=False)
                align_words = set()
                for i_sub, j_sub in align_subwords:
                    align_words.add((
                        item['sub2word_src'][i_sub.item()],
                        item['sub2word_tgt'][j_sub.item()]
                    ))
                
                # 计算分数
                src_words = {t[0] for t in align_words}
                tgt_words = {t[1] for t in align_words}
                n_src = len(src_words)
                n_tgt = len(tgt_words)
                
                precision = n_tgt / item['tgt_len'] if n_tgt > 0 else 0
                recall = n_src / item['src_len'] if n_src > 0 else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                align_score_list.append(f1)
    end = time.time()
    print(f"Alignment time: {end - mid:.2f} seconds")
    return align_score_list

def align_score4(srcs, tgts, model, tokenizer, batch_size=16):
    align_score_list = []
    align_layer = 24
    threshold = 1e-3

    # 检查是否有可用的GPU
    device = next(model.parameters()).device
    
    # Precompute all tokenizations and mappings
    tokenized_data = []
    
    for i in tqdm(range(len(srcs))):
        sent_src, sent_tgt = srcs[i], tgts[i]
        # Tokenize and convert to IDs
        token_src = [tokenizer.tokenize(word) for word in sent_src]
        token_tgt = [tokenizer.tokenize(word) for word in sent_tgt]
        
        wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
        wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        
        # Prepare for model - 使用更高效的方式
        ids_src = tokenizer.prepare_for_model(
            list(itertools.chain(*wid_src)), 
            return_tensors='pt', 
            truncation=True,
            max_length=tokenizer.model_max_length
        )['input_ids'].squeeze()
        
        ids_tgt = tokenizer.prepare_for_model(
            list(itertools.chain(*wid_tgt)), 
            return_tensors='pt', 
            truncation=True,
            max_length=tokenizer.model_max_length
        )['input_ids'].squeeze()
        
        # 使用列表推导式创建映射，更高效
        sub2word_map_src = [i for i, word_list in enumerate(token_src) for _ in word_list]
        sub2word_map_tgt = [i for i, word_list in enumerate(token_tgt) for _ in word_list]
        
        tokenized_data.append({
            'input_ids_src': ids_src,
            'input_ids_tgt': ids_tgt,
            'sub2word_src': sub2word_map_src,
            'sub2word_tgt': sub2word_map_tgt,
            'src_len': len(sent_src),
            'tgt_len': len(sent_tgt)
        })
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_data), batch_size)):
            batch = tokenized_data[i:i+batch_size]
            
            # Prepare batch inputs with pre-allocated tensors
            max_src_len = max(len(item['input_ids_src']) for item in batch)
            max_tgt_len = max(len(item['input_ids_tgt']) for item in batch)
            
            src_tensors = torch.full((len(batch), max_src_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            tgt_tensors = torch.full((len(batch), max_tgt_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            
            src_masks = torch.zeros((len(batch), max_src_len), dtype=torch.long, device=device)
            tgt_masks = torch.zeros((len(batch), max_tgt_len), dtype=torch.long, device=device)
            
            # 填充张量
            for j, item in enumerate(batch):
                src_len = len(item['input_ids_src'])
                tgt_len = len(item['input_ids_tgt'])
                
                src_tensors[j, :src_len] = item['input_ids_src'].to(device)
                tgt_tensors[j, :tgt_len] = item['input_ids_tgt'].to(device)
                
                src_masks[j, :src_len] = 1
                tgt_masks[j, :tgt_len] = 1
            
            # Get model outputs
            out_src = model(src_tensors, attention_mask=src_masks, output_hidden_states=True).hidden_states[align_layer]
            out_tgt = model(tgt_tensors, attention_mask=tgt_masks, output_hidden_states=True).hidden_states[align_layer]
            
            # Process each sentence in the batch
            for j, item in enumerate(batch):
                # 获取有效token的索引（排除[CLS]和[SEP]）
                src_len = len(item['input_ids_src'])
                tgt_len = len(item['input_ids_tgt'])
                
                # 直接使用切片获取有效表示，避免创建新张量
                valid_src = out_src[j, 1:src_len-1]  # Remove [CLS] and [SEP]
                valid_tgt = out_tgt[j, 1:tgt_len-1]
                
                # Calculate alignment with efficient matrix operations
                dot_prod = torch.matmul(valid_src, valid_tgt.transpose(-1, -2))
                
                # 使用log_softmax和exp组合，数值更稳定
                softmax_srctgt = torch.softmax(dot_prod, dim=-1)
                softmax_tgtsrc = torch.softmax(dot_prod, dim=-2)
                
                # 直接使用布尔运算，避免中间张量
                softmax_inter = (softmax_srctgt > threshold) & (softmax_tgtsrc > threshold)
                
                # 获取对齐的子词索引
                align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
                
                # 转换为词对齐
                align_words = set()
                for i_sub, j_sub in align_subwords:
                    align_words.add((
                        item['sub2word_src'][i_sub],
                        item['sub2word_tgt'][j_sub]
                    ))
                
                # Calculate scores
                src_words = {t[0] for t in align_words}
                tgt_words = {t[1] for t in align_words}
                n_src = len(src_words)
                n_tgt = len(tgt_words)
                
                precision = n_tgt / item['tgt_len'] if n_tgt > 0 else 0
                recall = n_src / item['src_len'] if n_src > 0 else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                align_score_list.append(f1)
    
    return align_score_list


# model_path = 'bert-base-multilingual-cased'
model_path = "/mnt/gemini/data1/yifengliu/model/bge-m3"
# model = transformers.BertModel.from_pretrained(model_path)
# tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

import hanlp
import hanlp_restful
from hanlp_restful import HanLPClient
# HanLP1 = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
# HanLP1 = hanlp.load(hanlp.pretrained.tok.UD_TOK_MMINILMV2L12)
# HanLP2 = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)


# src = "Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features."
# tgt = "由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。"
# tgt = "恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。"

src_lang = "eng"
tgt_lang = "deu"
src_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{src_lang}.devtest"
tgt_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{tgt_lang}.devtest"
src_dataset, tgt_dataset = load_flores(src_path), load_flores(tgt_path)

# import code; code.interact(local=locals())


# src_dataset = HanLP1(src_dataset)['tok']
# src_dataset = [[c for c in src if c not in [",", "\"", ".", '—', "(", ")", "/", "\\", "'"]]for src in src_dataset]
# tgt_dataset = HanLP2(tgt_dataset)
# tgt_dataset = [[c for c in tgt if c not in ["：", "。", "，", "“", "”", "（", "）", "·", "-", "/", "\\", "、"]]for tgt in tgt_dataset]
# src_dataset = ["Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days."]
# tgt_dataset = ["Dr. Ehud Ur, Professor von Medizin in Dalhousie Universität in Halifax, Nova Scotia und Stuhl von die klinisch und wissenschaftlich Division von die Kanadisch Diabetes Assoziation, vorsichtig dass die Forschung ist noch in seine frühe Tagen."]
# tgt_dataset = ["Den Dr. Ehud Earl, President vun der Clinical and Research Division vun der Canadian Diabetes Association a Professer op der Fakultéit fir Medizin vun der Dalhousie University zu Halifax, huet drop higewisen, datt dës Fuerschung nach an hire Virstadien ass a weider grëndlech Diskussioun brauch."]
# tgt_dataset = ["Den Dr. Ehud Earle, President vun der klinescher Fuerschung bei der Canadian Diabetes Association a Professer op der Dalhousie University School of Medicine zu Halifax, huet bemierkt, datt d'Fuerschung nach ëmmer am Ufank ass."]
# src_dataset = ["On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him."]
# tgt_dataset = ["Méindes huet d'Sara Danius, Sekretärin vum Nobelpräis fir Literaturcomité zu Stockholm a Schweden, an enger Sendung um schwedesche Sender Sveriges Radio ugekënnegt, datt de Comité decidéiert huet, opzeginn, de Kontakt mam Bob Dylan opzehuelen, fir ze bestätegen, ob hien tatsächlech den Nobelpräis fir Literatur 2016 gewonnen huet, well se hien net direkt kontaktéiere konnten. Virdrun hat de Comité schonn e puermol probéiert, den Dylan iwwer verschidde Mëttelen ze kontaktéieren, awer all Versich waren erfolleglos."]
# src_dataset = ["Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features."]
# tgt_dataset = ["Well Dinosaurierfiedere keng typesch Rachis hunn, de zentrale Schaft, deen duerch déi ganz Struktur vun der Fieder leeft, awer ëmmer nach déi grondleeënd Charakteristike vu Fiederen hunn, wéi z. B. Schläger a Filamenter, schléissen d'Fuerscher dovun of, datt d'Rachis-Struktur sech spéider graduell entwéckelt huet, während aner Eegeschafte wéi Schläger a Filamenter schonn fréier existéiert hunn."]
# tgt_dataset = ["Dinosaurierfiedere haten kee gutt entwéckelten Haaptstiel, genannt Rauchstiel, awer haten ëmmer nach aner Fiedercharakteristiken, wéi Barbelen a Barbelen, an d'Fuerscher schléissen dovun aus, datt d'Rauchstiel sech spéider wéi dës aner Charakteristiken entwéckelt huet."]

# src_dataset = ["\"We now have 4-month-old mice that are non-diabetic that used to be diabetic,\" he added."]
# tgt_dataset = ["\"Artı, 4 ay yaşındakı non-diabetik mişklər var ki, əvvəl diabetik olublar,\" deyib."]
# tgt_dataset = ["\"Hazırda bizim diabetdən əziyyət çəkməyən, amma əvvəl diabet olan 4 aylıq siçanlarımız var\", deyə o əlavə etdi."]

# src_dataset = ["On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him."]
# tgt_dataset = ["সোমবারে, সারা ডানিউস, স্বেডিশ একাডেমির লিটারেচার কমিটির প্রতিনিধি, স্বেডিশ রেডিও (Sveriges Radio) এর একটি রেডি প্রোগ্রামে ঘোষণা করেন যে কমিটি বোব ডালনকে ডাকার চেষ্টা করেছিল কিন্তু তাঁর সাথে যোগাযোগ করতে সক্ষম হতে পারেনি, তাই তাঁর সাথে যোগাযোগের চেষ্টা বন্ধ করে দেয়া হয়েছে। 2016 সালের লিটারেচার নোবেল পুরস্কারের জন্য এই কমিটি তাঁকে প্রাপ্ত হও য়ার বিষয়ে তাঁর সাথে যোগাযোগ করার চেষ্টা করেছিল কিন্তু সম্ভবত তা সম্পূর্ণ অসম্ভব হয়ে গেছে।"]
# tgt_dataset = ['সোমবার, সুইডেনের সেভিরস রেডিওতে একটি অনুষ্ঠানের সময় সুইডিশ অ্যাকাডেমিতে সাহিত্যের নোবেল কমিটির স্থায়ী সচিব, সারা ডনিয়াস,  প্রকাশ্যে ঘোষণা করেছিলেন যে কমিটি, ২০১৬ সালে সাহিত্যে নোবেল পুরস্কার পাওয়ার বিষয়ে বব ডিলানের সঙ্গে সরাসরিযোগাযোগ করতে না পেরে, তাঁর কাছে পৌঁছানোর প্রচেষ্টা ত্যাগ করেছিল।']
src_dataset = [src.split() for src in src_dataset]
tgt_dataset = [tgt.split() for tgt in tgt_dataset]
# src_dataset, tgt_dataset = src_dataset[-1:], tgt_dataset[-1:]  # for test
# align_score_list = align_score(src_dataset, tgt_dataset, model, tokenizer)
align_score_list = align_score2(src_dataset, tgt_dataset, model, tokenizer, batch_size=16)
# align_score_list = align_score4(src_dataset, tgt_dataset, model, tokenizer, batch_size=16)
# align_score_list = align_score3(src_dataset, tgt_dataset, model, tokenizer, batch_size=1024)
import code; code.interact(local=locals())
# myaligner = SentenceAligner(model="bge-m3", token_type="bpe", matching_methods="mai")

# # The source and target sentences should be tokenized to words.
# # trg_sentence = "Pracovníci musejí získat schválení nadřízených ohledně každého svého rozhodnutí a očekává se od nich, že instrukce svých nadřízených uposlechnou bez otázek."
# # src_sentence = "Workers must often get their superiors' approval for any decisions they make, and are expected to obey their superiors' instructions without question."
# align_score_list = []
# for i in tqdm(range(len(src_dataset))):
#   src_sentence = src_dataset[i]
#   trg_sentence = tgt_dataset[i]
  
#   alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)
#   # import code; code.interact(local=locals())
#   src_words = set({t[0]: t for t in sorted(alignments['mwmf'])}.values())
#   tgt_words = set({t[1]: t for t in sorted(alignments['mwmf'])}.values())

#   precision = min(len(tgt_words) / len(trg_sentence), 1)
#   recall = min(len(src_words) / len(src_sentence), 1)
#   f1 = 2 * precision * recall / (precision + recall)
#   print_alignments(src_sentence, trg_sentence, alignments['mwmf'])
#   # import code; code.interact(local=locals())
#   # print_alignments(src_sentence, trg_sentence, alignments['itermax'])
#   # import code; code.interact(local=locals())
#   if f1 < 0.9:
#     import code; code.interact(local=locals())
#   align_score_list.append(f1)
#   print(f1)

# for matching_method in alignments:
#     print(matching_method, ":", alignments[matching_method])

# Expected output:
# mwmf (Match): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
# inter (ArgMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
# itermax (IterMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


# tokenizer = AutoTokenizer.from_pretrained("/mnt/gemini/data1/yifengliu/model/bge-m3")
# model = AutoModel.from_pretrained("/mnt/gemini/data1/yifengliu/model/bge-m3")


# from simalign import SentenceAligner
# import jieba
# # making an instance of our model.
# # You can specify the embedding model and all alignment settings in the constructor.
# myaligner = SentenceAligner(model="bge-m3", token_type="bpe", matching_methods="mai")
# tokenizer = myaligner.embed_loader.tokenizer
# # The source and target sentences should be tokenized to words.
# # src_sentence = "This is a test ."
# # trg_sentence = "Das ist ein Test ."
# align_score_list = []
# # src_dataset = ["This is a test."]
# # tgt_dataset = ["这是一个测试。"]
# # import code; code.interact(local=locals())
# for i in tqdm(range(len(src_dataset))):
# # for src_sentence, trg_sentence in zip(src_dataset, tgt_dataset):
#   src_sentence = src_dataset[i].strip()
#   src_sentence = src_sentence.split()
#   trg_sentence = tgt_dataset[i].strip()
#   trg_sentence = list(jieba.cut(trg_sentence))
#   trg_sentence = [trg for trg in trg_sentence if trg.strip() != '']
#   # src_sentence = tokenizer.tokenize(src_sentence)
#   # trg_sentence = tokenizer.tokenize(trg_sentence)
  
#   # src_sentence = ["This", "is", "a", "test", "."]
#   # trg_sentence = ["Das", "ist", "ein", "Test", "."]
#   # import code; code.interact(local=locals())
#   # The output is a dictionary with different matching methods.
#   # Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
#   alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)
#   # alignment_len = max(len(alignments[method]) for method in alignments)
#   alignment_len = len(alignments['itermax'])
#   alignment_score = alignment_len / len(trg_sentence)
#   align_score_list.append(alignment_score)

#   print(src_dataset[i])
#   print(tgt_dataset[i])
#   for matching_method in alignments:
#       # alignments[matching_method] = [(src_idx, trg_idx) for src_idx, trg_idx in alignments[matching_method] if src_idx < len(src_sentence) and trg_idx < len(trg_sentence)]
#       alignment_list = alignments[matching_method]
#       fixed_list = []
#       # import code; code.interact(local=locals())
#       for alignment_pair in alignment_list:
#           src_idx, trg_idx = alignment_pair
#           print(src_idx, trg_idx)
#           # import code; code.interact(local=locals())
#           fixed_list.append((src_sentence[src_idx], trg_sentence[trg_idx]))
#       print(matching_method, ":", fixed_list, len(fixed_list))


plt.hist(align_score_list, bins=[i/10 for i in range(11)], edgecolor='black')

plt.xlabel('Value Range')
plt.ylabel('Count')
plt.title('F1 Score Distribution')
plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
plt.savefig(f"/mnt/gemini/data1/yifengliu/qe-lr/output/{tgt_lang}3.png")
print(f"Align Score: {align_score_list}")

import code; code.interact(local=locals())