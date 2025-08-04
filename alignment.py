import json
import torch
import transformers
import itertools
import jieba
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tqdm import *
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

def print_alignments(src, tgt, alignments):
  for alignment_pair in alignments:
    src_idx, trg_idx = alignment_pair
    print(f"{src[src_idx]} --> {tgt[trg_idx]}")

def align_score(srcs, tgts, model, tokenizer):
  align_score_list = []
  # for src, tgt in zip(srcs, tgts):
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
    align_percent = len(align_words) / len(token_tgt)
    align_score_list.append(align_percent if align_percent <= 1 else 1)
  return align_score_list

model_path = 'bert-base-multilingual-cased'
# model_path = "/mnt/gemini/data1/yifengliu/model/bge-m3"
model = transformers.BertModel.from_pretrained(model_path)
tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
# model = BGEM3FlagModel(model_path)
# model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer = 
# import code; code.interact(local=locals())
# src = "Because the dinosaur feathers do not have a well-developed shaft, called a rachis, but do have other features of feathers — barbs and barbules — the researchers inferred the rachis was likely a later evolutionary development that these other features."
# tgt = "由于恐龙羽毛缺乏典型的羽毛轴（rachis），即羽毛中贯穿整个结构的中轴部分，但仍然具备羽毛的基本特征，如羽片和羽丝，研究人员据此推断，羽毛轴这一结构可能是后来才逐渐演化出来的，而羽片和羽丝等其他特征则可能在更早的时候就已经存在了。"
# tgt = "恐龙的羽毛并没有发育良好的主干——这称为“羽轴”，但还是有羽毛的其他特征，比如羽枝和羽小枝，研究人员推断羽轴的进化可能比这些其他特征晚。"

src_lang = "eng"
tgt_lang = "zho_simpl"
src_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{src_lang}.devtest"
tgt_path = f"/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest/{tgt_lang}.devtest"
src_dataset, tgt_dataset = load_flores(src_path), load_flores(tgt_path)
# dataset = my_load_dataset("/mnt/gemini/data1/yifengliu/qe-lr/output/flores/Rule-Detect-MetricX-Qwen2.5-0.5B-en-zh-1M-bsz128/global_step780_hf/eng-zho_simpl.txt")
src_dataset = src_dataset[:100]
tgt_dataset = tgt_dataset[:100]
# src_dataset = [data['src'] for data in dataset]
# tgt_dataset = [data['pred'] for data in dataset]
src_dataset = [src.strip().split() for src in src_dataset]
tgt_dataset = [list(jieba.cut(tgt.strip())) for tgt in tgt_dataset]
tgt_dataset = [[t for t in tgt if len(t.strip()) > 0 and "：" not in t and "，" not in t and "。" not in t and "“" not in t and "”" not in t and "(" not in t and ")" not in t] for tgt in tgt_dataset]
# import code; code.interact(local=locals())
# for src, tgt in zip(src_dataset, tgt_dataset):
align_score_list = align_score(src_dataset, tgt_dataset, model, tokenizer)
# tokenizer = AutoTokenizer.from_pretrained("/mnt/gemini/data1/yifengliu/model/bge-m3")
# model = AutoModel.from_pretrained("/mnt/gemini/data1/yifengliu/model/bge-m3")


# src_dataset = [src.strip() for src in src_dataset]
# tgt_dataset = [" ".join(list(jieba.cut(tgt.strip()))) for tgt in tgt_dataset]
# align_score_list = align_score(src_dataset, tgt_dataset, model, tokenizer)


# from FlagEmbedding import BGEM3FlagModel

# model = BGEM3FlagModel('/mnt/gemini/data1/yifengliu/model/bge-m3',  use_fp16=True) 

# sentences_1 = ["Siminoff said sales boosted after his 2013 appearance in a Shark Tank episode where the show panel declined funding the startup."]
# sentences_2 = ["Siminoof akka jedhutti erga inni 2013 kutaa fiilmii Taankii Shaark iddoo itti beeksisni agarsiisichaa deggersa jalqabsiisuu kuffise’en booda gurgurtaan baayyee dabale jedhe.", 
#                "西米诺夫说，2013 年他在《创智赢家》节目中露面后，公司的销售额大增，当时节目组拒绝向这家初创公司投资。",
#                "Siminoff alisema mauzo yaliongezeka baada ya yeye kuonekana katika kipindi cha Shark Tank mnamo 2013 ambapo paneli ya onyesho hilo ilikataa kufadhili biashara hiyo mpya.",
#                ]

# sentences_1 = ["USA Gymnastics supports an independent investigation that may shine light on how abuse of the proportion described so courageously by the survivors of Larry Nassar could have gone undetected for so long and embraces any necessary and appropriate changes."]
# sentences_2 = ["Jiimnaastikiin USA qorannoo of danda’ee deggera sunis yaanni sabboonummaan kan Leerii Naasaar lubbun isa hafeen kan kenname hammam badaa akka kan sirriitti mul’isa, kunimmo kan yeroo dheeraaf hin beekamin turee, osoo hin haammatamin turefi jijjiirran sirriin kan hin kennamneefidha.",
#                "根据美国体操协会支持的一项独立调查，我们也许能够得知，幸存者勇敢曝光的、拉里·纳萨尔的大规模性侵行为，为什么在这么长时间内都没有被发现。此外，协会还表示会做出必要和适当的整改。",
#                "USA Gymnastics inaunga mkono uchunguzi huru ambao labda utafafanua jinsi unyanyasaji wa kiwango kama vile ulivyoelezewa kwa ujasiri sana na manusura wa Larry Nassar ungekosa kugunduliwakwa muda mrefu hivyo na inakumbatia mabadiliko muhimu na ya kufaa."]


# sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]

# print(model.compute_score(sentence_pairs, 
#                           max_passage_length=128, # a smaller max length leads to a lower latency
#                           weights_for_different_modes=[0.4, 0.2, 0.4])) # weights_for_different_modes(w) is used to do weighted sum: w[0]*dense_score + w[1]*sparse_score + w[2]*colbert_score



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
plt.title('Histogram with 0.1 bins')
plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
plt.savefig(f"{tgt_lang}.png")
print(f"Align Score: {align_score_list}")

import code; code.interact(local=locals())