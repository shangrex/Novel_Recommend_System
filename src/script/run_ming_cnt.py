import spacy
from spacy.lang.zh.examples import sentences 
import argparse
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


nlp = spacy.load("zh_core_web_lg")
nlp.enable_pipe("senter")

parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--txt', type=str, required=True,
                    help='the word for searching')

parser.add_argument('--topk', type=int, required=False,
                    default=10,
                    help='# of results')

args = parser.parse_args()

topk = args.topk

ming = pd.read_csv('data/mingyan_new.csv')


f = open(f'data/pretrain/spa_ming_emb.pkl', 'rb')
spa_emb = pickle.load(f)

doc = nlp(args.txt)

rst = []

for i in tqdm(range(len(spa_emb))):
    ftmp = cosine_similarity([doc.vector], [spa_emb[i][3]])
    rst.append([ftmp, ming['paragraphs'].iloc[i], ming['author'].iloc[i], ming['name'].iloc[i]])


rst = sorted(rst, key=lambda i : i[0], reverse=True)

for i in range(topk):
    print(rst[i][0][0][0])
    print(rst[i][1:])

