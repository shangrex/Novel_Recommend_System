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
from utils import io_file as io

from tknzr._jieba import jiebatknzr 
from tknzr._char import chartknzr
from tknzr._base import basetknzr
from tknzr._whitespace import wstknzr
from utils.model import one_hot
from dset._ch_poem import chpoemdset
from model._bert import transformers_bert
import torch 
from torch.nn.functional import softmax
from transformers import *

from model._base import basemodel
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from torch.utils.data import DataLoader
from tqdm import tqdm


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

# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=10)
# print(model)
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# print(inputs)
# print(labels)
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits

# model.train()
# inputs = tokenizer("Hello I am Rex", return_tensors="pt")
# labels = torch.tensor([9]).unsqueeze(0)
# outputs = model(**inputs, labels=labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-chinese"
tokenizer =  AutoTokenizer.from_pretrained(model_name)

        
poem_dset = chpoemdset("train", tokenizer)
# x = poem_dset[0]
# print(x[1])
# for i in x:
#     print(type(i))
# print(len(poem_dset))
# print(poem_dset.dtknz(x[0]))

BATCH_SIZE = 4
trainloader = DataLoader(poem_dset, batch_size=BATCH_SIZE, 
                         collate_fn=poem_dset.create_mini_batch)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=poem_dset.len_lbl())
print("model")



def get_valid_prediction(model):
    acc = 0
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(trainloader):
            
            data = [i.to(device) for i in data]

            token_tensor , mask_tensor, lbl_tensor = data[:3]
            # print(token_tensor.shape)
            # for i in token_tensor:
            #     print(poem_dset.dtknz(i))
            # print(mask_tensor.shape)
            # print(lbl_tensor)
            outputs = model(input_ids = token_tensor,
                            attention_mask = mask_tensor)

            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            lbl_tensor = torch.argmax(lbl_tensor)
            total += len(pred)
            correct += (lbl_tensor == pred).sum().item()
   

        print(correct, total)
        acc = correct / total
    return acc

def get_train(model, epoch):
    model.train()
    total_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for e in range(epoch):
        running_loss = 0
        for data in tqdm(trainloader):
            data = [i.to(device) for i in data]
            optimizer.zero_grad()

            token_tensor , mask_tensor, lbl_tensor = data[:3]
            # print(token_tensor.shape)
            # print(lbl_tensor)
            # print(lbl_tensor.shape)
            lbl_tensor = torch.argmax(lbl_tensor, 1)
            # print(lbl_tensor)
            outputs = model(input_ids = token_tensor,
                            attention_mask = mask_tensor,
                            labels = lbl_tensor)

            # print("optimizer start...")
            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
        
        print(running_loss)
        total_loss += running_loss
    return total_loss / epoch
    

# print("device:", device)
# model = model.to(device)
# print("valid testing ....")
# acc = get_valid_prediction(model)
# print("classification acc:", acc)

# print("trainning....")
# model = model.to(device)
# avg_loss = get_train(model, 1)
# print("avg loss", avg_loss)

# model.save_pretrained('data/exp/1/model.pth')
# config.save('data/exp/1/config.json')