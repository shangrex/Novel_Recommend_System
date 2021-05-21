'''
train the mingyan with spacy
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

ming = pd.read_csv('data/mingyan_new.csv')


rst = []


rst = []

for i in tqdm(range(len(ming))):
    tmp = nlp(ming['paragraphs'].iloc[i])
    ftmp = tmp.vector
    rst.append([ming['paragraphs'].iloc[i], ming['author'].iloc[i], ming['name'].iloc[i], ftmp])


with open(f'data/pretrain/spa_ming_emb.pkl', 'wb') as f:
    pickle.dump(rst, f)



