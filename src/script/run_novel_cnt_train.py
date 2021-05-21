'''
Train the most similar novel from the input words through spacy (Word Embedding)
'''
import spacy
from spacy.lang.zh.examples import sentences 
import argparse
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


nlp = spacy.load("zh_core_web_lg")
nlp.enable_pipe("senter")


nov = pd.read_csv('data/novel.csv')


rst = []

for i in tqdm(range(len(nov))):
    tmp = nlp(nov['paragraphs'].iloc[i])
    ftmp = tmp.vector
    rst.append([nov['name'].iloc[i], nov['author'].iloc[i], nov['paragraphs'].iloc[i], ftmp])


with open(f'data/pretrain/spa_nov_emb.pkl', 'wb') as f:
    pickle.dump(rst, f)