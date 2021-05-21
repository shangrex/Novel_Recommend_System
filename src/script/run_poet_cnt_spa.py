'''
Use Spacy (Word Embedding) to Recommend Poet
'''
import spacy
from spacy.lang.zh.examples import sentences 
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


nlp = spacy.load("zh_core_web_lg")
nlp.enable_pipe("senter")

parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--txt', type=str, required=True,
                    help='the word for searching')

parser.add_argument('--topk', type=int, required=False,
                    default=100,
                    help='# of results')

args = parser.parse_args()

topk = args.topk


f = open(f'data/pretrain/spa_embedding.pkl', 'rb')
spa_emb = pickle.load(f)

poet = pd.read_csv('data/poet.csv')
doc = nlp(args.txt)

rst = []


for i in tqdm(range(len(spa_emb))):
    # print(i[3].dtype)
    # print(doc.vector.dtype)
    ftmp = cosine_similarity([spa_emb[i][3]], [doc.vector])
    rst.append([ftmp[0][0], poet['title'].iloc[i], poet['author'].iloc[i], poet['paragraphs'].iloc[i]])


rst = sorted(rst, key=lambda i : i[0], reverse=True)


cnt_rst = {}
for i in range(topk):
    if rst[i][1] in cnt_rst:
        cnt_rst[rst[i][1]+rst[i][2]] += 1
    else:
        cnt_rst[rst[i][1]+rst[i][2]] = 1
        
cnt_rst = sorted(cnt_rst.items(), key=lambda i: i[1], reverse=True)

for i in cnt_rst:
    print(i)