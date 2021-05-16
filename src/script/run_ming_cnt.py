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
                    default=10,
                    help='# of results')

args = parser.parse_args()

topk = args.topk

ming = pd.read_csv('data/mingyan_new.csv')

doc = nlp(args.txt)

rst = []

for i in tqdm(range(len(ming))):
    tmp = nlp(ming['paragraphs'].iloc[i])
    ftmp = doc.similarity(tmp)
    rst.append([ftmp, ming['author'].iloc[i], ming['paragraphs'].iloc[i], ming['name'].iloc[i]])


rst = sorted(rst, key=lambda i : i[0], reverse=True)

for i in range(topk):
    print(rst[i][0])
    print(rst[i][1:])

