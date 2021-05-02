'''
search from topic relative
'''

import pickle
from tqdm import tqdm
import pandas as pd 
import argparse

poet = pd.read_csv('data/poet.csv')

parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--find', type=str, required=True,
                    help='the word for searching')

parser.add_argument('--min', required=False, type=int, default=3,
                    help="the threshold of match count")

args = parser.parse_args()

target = args.find.split(' ')
min_count = args.min

f = open('data/topic/1000_30topic.pkl', 'rb')
top_dict = pickle.load(f)


print("=="*7+"finding topic"+"==")
match_count = 0
rst = {}
for i, j in top_dict.items():
    for k in j:
        for t in target:
            if k in t:
                match_count += 1
    rst[i] = match_count
    match_count = 0

for i in rst:
    print(i)

# print("=="*7+"topic relative"+"==")
# for i in tqdm(range(len(poet))):
#     txt = poet['paragraphs'].iloc[i]
#     for j, k in top_dict.items():
#         print(j, k)
#         break
#     break
