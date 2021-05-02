'''
Train the most similar poet from the input words through spacy (Word Embedding)
'''
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import *
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.zh.examples import sentences 
import pandas as pd
import os
import argparse
import torch 


parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--txt', type=str, required=True, 
                    help='the input sentence')

parser.add_argument('--topk', type=int, required=False, default=5, 
                    help='# of result')

args = parser.parse_args()
txt = args.txt
topk = args.topk

poet = pd.read_csv('data/poet.csv')

# txt = "國破山河在，城春草木深。"
# txt = "舉杯邀明月，對影成三人。"
# txt = "天若不愛酒，酒星不在天。"

nlp = spacy.load("zh_core_web_sm")
nlp.enable_pipe("senter")
doc = nlp(txt)

rst = {}
for i in tqdm(range(len(poet))):
    tmp_doc = nlp(poet['paragraphs'].iloc[i])
    rst[i] = doc.similarity(tmp_doc[0:len(doc.text)-1])

sorted(rst.itmes(), key=lambda item: item[1])


count = 0
for i, j in rst.items():
    count += 1
    if count > topk:
        break

    print(poet['author'].iloc[i], j)