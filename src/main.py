import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
import os

from src.tknzr._jieba import jiebatknzr 
from src.tknzr._char import chartknzr
from src.tknzr._base import basetknzr
from src.tknzr._whitespace import wstknzr
from src.utils.model import one_hot
from src.dset._ch_poem import chpoemdset
from src.model._bert import transformers_bert
import torch 
from torch.nn.functional import softmax
from transformers import *

from src.model._base import basemodel
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import *


# j = jiebatknzr()
# s = """我是一条天狗呀！
# 我把月来吞了，
# 我把日来吞了，
# 我把一切的星球来吞了，
# 我把全宇宙来吞了。
# 我便是我了！"""


# corpus = [
#   "帮我 查下 明天 北京 天气 怎么样",
#   "帮我 查下 今天 北京 天气 好不好",
#   "帮我 查询 去 北京 的 火车",
#   "帮我 查看 到 上海 的 火车",
#   "帮我 查看 特朗普 的 新闻",
#   "帮我 看看 有没有 北京 的 新闻",
#   "帮我 搜索 上海 有 什么 好玩的",
#   "帮我 找找 上海 东方明珠 在哪"
# ]


# w = [["乾坤運轉是尋常，人有依違自短長。",
#       "不覺貪生身外苦，如駒過隙百年光。"],
#       ["常思清凈世情閑，遥認天平我扣關。",
#       "莫道玄談些子是，鳳飛鶴宿九疑山。"]]

# data = pd.DataFrame()
# c_tknz = chartknzr()  
# data['paragraphs'] = w
# for i in data['paragraphs']:
# 	for j in i:
# 		c_tknz.build_vocab(j)
# 		x = c_tknz.tknz(j)
# 		y = c_tknz.enc(x)
		
# 		# print(pad_sequence(y, 7))
# 		# print(y)		

# model = transformers_bert('voidful/albert_chinese_tiny')
# data = pd.read_csv('data/poet.csv')
# transformers_bert('hfl/chinese-roberta-wwm-ext-large')
# inputtext = "今天[MASK]情很好"
# transformers_bert.enc(inputtext)

# input_ids = torch.tensor(transformers_bert.enc(inputtext)).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids, masked_lm_labels=input_ids)
# loss, prediction_scores = outputs[:2]
# logit_prob = softmax(prediction_scores[0, maskpos]).data.tolist()
# predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# print(predicted_token,logit_prob[predicted_index])



# pretrained = 'hfl/chinese-roberta-wwm-ext-large'
# tokenizer = BertTokenizer.from_pretrained(pretrained)
# model = BertModel.from_pretrained(pretrained)

# inputtext = "[CLS] 當時婦棄夫，今日夫棄婦。, 若不逞丹青，空房應獨守。[MASK]"
# x = tokenizer.encode(inputtext)
# print(x)
# model.eval()
# with torch.no_grad():
# 	outputs = model


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

text = "[MASK]離海底千山黑，纔到天中萬國明。"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

from transformers import BertForMaskedLM

# 除了 tokens 以外我們還需要辨別句子的 segment ids
# tokens_tensor = torch.LongTensor([ids])  # (1, seq_len)
# segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
# maskedLM_model = BertForMaskedLM.from_pretrained('data/pretrain/bert-base-chinese/1/')
# maskedLM_model = BertForMaskedLM.from_pretrained('bert-base-chinese')

# 使用 masked LM 估計 [MASK] 位置所代表的實際 token 
# maskedLM_model.eval()
# with torch.no_grad():
#     outputs = maskedLM_model(tokens_tensor, segments_tensors)
#     predictions = outputs[0]
#     # (1, seq_len, num_hidden_units)
# del maskedLM_model

# # 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
# masked_index = 0
# k = 3
# probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
# predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

# # 顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
# print("輸入 tokens ：", tokens)
# print('-' * 50)
# for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
#     tokens[masked_index] = t
#     print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokens), '...')



# from src.dset._ch_poem import chpoemdset
# from transformers import *
# tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')
# poem_train = chpoemdset("train", tokenizer, 0)
# poem_test = chpoemdset("test", tokenizer, 0)
# poem_dset = chpoemdset("all", tokenizer, 0)

# print("all data", len(poem_dset))
# print("train data",len(poem_train))
# print("test data", len(poem_test))

# print(poem_dset[11])
# print(poem_dset.dtknz(poem_dset[11][0]))
# print(poem_dset.dauthor(poem_dset[11][1]))


# from src.dset._ch_poem import chpoemdset_tag
# from transformers import *
# tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')
# tags_train = chpoemdset_tag("train", tokenizer, 0)
# tags_test = chpoemdset_tag("test", tokenizer, 0)
# tags_dset = chpoemdset_tag("all", tokenizer, 0)

# print("all data", len(tags_dset))
# print("train", len(tags_train))
# print("test", len(tags_test))


from src.dset._ch_poem import chpoemdset_gen_author
from transformers import *
from transformers import GPT2Tokenizer, GPT2Model
model = GPT2Model.from_pretrained('ckiplab/gpt2-base-chinese')
tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')
gen_poem = chpoemdset_gen_author(tokenizer, "李白", 56)


print(gen_poem[0])
print(gen_poem.dtknz(gen_poem[0][0]))
print(gen_poem.dtknz(gen_poem[0][1]))
