'''
Implement a search engine.
'''
import pickle
from tqdm import tqdm
import pandas as pd 
import argparse


parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--find', type=str, required=True,
                    help='the word for searching')
parser.add_argument('--min', required=False, type=int, default=1,
                    help="the threshold of match count")

args = parser.parse_args()

target = args.find.split(' ')
print(target)
min_count = args.min

poet = pd.read_csv('data/poet.csv')


rst_atr = []
print("=="*7+"searching author"+"=="*7)
for i in tqdm(range(len(poet))):
    for t in target:
        if poet['author'].iloc[i] == t:
            rst_atr.append({t:poet['paragraphs'].iloc[i]})

rst_tit = []
print("=="*7+"searching title"+"=="*7)
for i in tqdm(range(len(poet))):
    for t in target:
        if type(poet['title'].iloc[i]) == float:
            continue
        if t in poet['title'].iloc[i]:
            rst_tit.append({poet['title'].iloc[i]+" "+poet['author'].iloc[i]:poet['paragraphs'].iloc[i]})


print("=="*7+"searching paragraphs"+"==")
rst_cnt = []
for i in tqdm(range(len(poet))):
    check_count = 0
    for t in target:
        if t in poet['paragraphs'].iloc[i]:
            check_count += 1
    if check_count > 0:
        rst_cnt.append({'match_count':check_count, poet['author'].iloc[i]:poet['paragraphs'].iloc[i]})
#sort 
sorted(rst_cnt, key=lambda x: x['match_count'])

print("=="*7+"result author"+"=="*7)
for i in rst_atr:
    print(i)

print("=="*7+"result title"+"=="*7)
for i in rst_tit:
    print(i)

print("=="*7+"result content"+"=="*7)
for i in rst_cnt:
    if i['match_count'] < min_count:
        break
    for j, k in i.items():
        print(j, k)




