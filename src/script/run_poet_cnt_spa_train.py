'''
Train the most similar poet from the input words through spacy (Word Embedding)
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


poet = pd.read_csv('data/poet.csv')


rst = []

for i in tqdm(range(len(poet))):
    tmp = nlp(poet['paragraphs'].iloc[i])
    ftmp = tmp.vector
    rst.append([poet['title'].iloc[i], poet['author'].iloc[i], poet['paragraphs'].iloc[i], ftmp])


with open(f'data/pretrain/spa_embedding.pkl', 'wb') as f:
    pickle.dump(rst, f)