import json
import torch
import transformers
import itertools
import jieba
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tqdm import *
from collections import defaultdict
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
      print(f'{color.BOLD}{color.BLUE}{src[src_idx]}{color.END}==={color.BOLD}{color.RED}{tgt[trg_idx]}{color.END}')
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
    align_layer = 8
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
    # print_alignments(sent_src, sent_tgt, align_words)
    # import code; code.interact(local=locals())
    f1_list.append(f1)
  return f1_list
  # return align_score_list

model_path = 'bert-base-multilingual-cased'
# model_path = "/mnt/gemini/data1/yifengliu/model/bge-m3"
model = transformers.BertModel.from_pretrained(model_path)
tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

import hanlp
import hanlp_restful
from hanlp_restful import HanLPClient
HanLP1 = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
# HanLP1 = hanlp.load(hanlp.pretrained.tok.UD_TOK_MMINILMV2L12)
HanLP2 = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)


# model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# src = "Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features."
# tgt = "由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。"
# tgt = "恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。"

src_lang = "eng"
tgt_lang = "zho_simpl"
src_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{src_lang}.devtest"
tgt_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{tgt_lang}.devtest"
src_dataset, tgt_dataset = load_flores(src_path), load_flores(tgt_path)
# src_dataset, tgt_dataset = src_dataset[13:23], tgt_dataset[13:23]
# src_dataset, tgt_dataset = src_dataset[-2:], tgt_dataset[-2:]

# src_dataset = ["Duvall, who is married with two adult children, did not leave a big impression on Miller, to whom the story was related."]
# tgt_dataset = ["杜瓦尔已婚，有两个成年子女，但未给米勒留下深刻印象。"]
# tgt_dataset = ["杜瓦尔已婚，有两个已经成年的孩子，他并没有给故事的讲述者米勒留下太大印象。"]
# src_dataset = ["Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features."]
# tgt_dataset = ["由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。"]
# tgt_dataset = ["恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。"]
src_dataset = ["Officials for the city of Amsterdam and the Anne Frank Museum state that the tree is infected with a fungus and poses a public health hazard as they argue that it was in imminent danger of falling over."
               ]
tgt_dataset = ["阿姆斯特丹市政府及安妮·弗兰克博物馆的工作人员表示，这棵橡树已被真菌感染，存在安全隐患，随时可能倒下，对公众安全构成威胁。因此，他们主张应尽快采取措施处理这棵树。"
               ]

src_dataset = HanLP1(src_dataset)['tok']
# import code; code.interact(local=locals())
src_dataset = [[c for c in src if c not in [",", "\"", ".", '—', "(", ")", "/", "\\", "'"]]for src in src_dataset]
# import code; code.interact(local=locals())
tgt_dataset = HanLP2(tgt_dataset)
# import code; code.interact(local=locals())
tgt_dataset = [[c for c in tgt if c not in ["：", "。", "，", "“", "”", "（", "）", "·", "-", "/", "\\", "、"]]for tgt in tgt_dataset]
# import code; code.interact(local=locals())
# for src, tgt in zip(src_dataset, tgt_dataset):
align_score_list = align_score(src_dataset, tgt_dataset, model, tokenizer)
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
plt.savefig(f"{tgt_lang}.png")
print(f"Align Score: {align_score_list}")

import code; code.interact(local=locals())