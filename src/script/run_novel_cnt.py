import spacy
from spacy.lang.zh.examples import sentences 
import argparse
import pandas as pd
import numpy as np
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

nov = pd.read_csv('data/mingyan_new.csv')

doc = nlp(args.txt)

rst = []

for i in tqdm(range(len(nov))):
    tmp = nlp(nov['paragraphs'].iloc[i])
    ftmp = doc.similarity(tmp)
    rst.append([ftmp, nov['name'].iloc[i], nov['author'].iloc[i], nov['paragraphs'].iloc[i]])


rst = sorted(rst, key=lambda i : i[0], reverse=True)


cnt_rst = {}
for i in range(topk):
    if rst[i][1] in cnt_rst:
        cnt_rst[rst[i][1]] += 1
    else:
        cnt_rst[rst[i][1]] = 1
        
cnt_rst = sorted(cnt_rst.items(), key=lambda i: i[1], reverse=True)

for i in cnt_rst:
    print(i)