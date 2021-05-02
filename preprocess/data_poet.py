#make the novel data
# from read_file import XhtmlParser
from tqdm import tqdm
import pandas as pd
import jieba
import os

#poet text classification
poet = pd.DataFrame()
#read song poet 255, 57
#read song poet
for i in tqdm(range(255)):
    f = pd.read_json('data/json/poet.song.{}.json'.format(i*1000))
    poet = poet.append(f)
#read poet song strains
strains = pd.DataFrame()
for i in tqdm(range(255)):
    f = pd.read_json('data/strains/json/poet.song.{}.json'.format(i*1000))
    strains = strains.append(f)

#used to be 57
for i in tqdm(range(58)):
    f = pd.read_json('data/json/poet.tang.{}.json'.format(i*1000))
    poet = poet.append(f)
#read tang poet's strains
for i in tqdm(range(58)):
    f = pd.read_json('data/strains/json/poet.tang.{}.json'.format(i*1000))
    strains = strains.append(f)

poet = pd.merge(poet, strains, on="id")

#transfer paragraphs from list to str
print("=="*7+"transfer paragraphs to str"+"=="*7)
for i in tqdm(range(len(poet))):
    s = ""
    for j in poet['paragraphs'].iloc[i]:
        s += j
    poet['paragraphs'].iloc[i] = s

#count poet length
print("=="*7+"counting paragraphs"+"=="*7)
count = 0
poet_len = []
last = []
for i in tqdm(poet['paragraphs']):
    p_len = 0
    for j in i:
        p_len += len(j)
        if '。' in j:
            p_len -= 1
        if '，' in j:
            p_len -= 1
    if len(i) == 0:
        poet_len.append(0)
    else:
        poet_len.append(p_len)
    count += 1
    last = i
print("# of count len", count)
poet['len'] = poet_len

print("=="*7+"adding trend data on poet"+"=="*7)
hot_data = pd.DataFrame()
#0~25400
for i in tqdm(range(255)):
    f = pd.read_json('data/rank/poet/poet.song.rank.{}.json'.format(i*1000))
    hot_data = hot_data.append(f)
#0~57
for i in tqdm(range(58)):
    f = pd.read_json('data/rank/poet/poet.tang.rank.{}.json'.format(i*1000))
    hot_data = hot_data.append(f)



poet['baidu'] = hot_data['baidu'].to_numpy()
poet['so360'] = hot_data['so360'].to_numpy()
poet['bing'] = hot_data['bing'].to_numpy()
poet['bing_en'] = hot_data['bing_en'].to_numpy()
poet['google'] = hot_data['google'].to_numpy()
poet['rank'] = (hot_data['baidu']+hot_data['so360']+hot_data['bing']+hot_data['bing_en']+hot_data['google']).to_numpy() 

print("=="*7+"dropping empty paragraphs"+"=="*7)
#drop empty paragraphs
empty_para = []
for i in poet['paragraphs']:
    if type(i) == float:
        empty_para.append(False)
    else:
        empty_para.append(True)
    count += 1
poet = poet.loc[empty_para]

print("=="*7+"tokenizing paragraphs"+"=="*7)
#add tokenize paragraphs
fstp = open('data/stopwords/_ch_poem.txt', 'r')
stopwords = []
for i in fstp.read():
    stopwords.append(i)

filter_poet = []
for i in tqdm(poet['paragraphs']):
    i = ''.join(list(filter(lambda x: x not in stopwords, i)))
    i = " ".join(list(jieba.lcut(i)))
    filter_poet.append(i)

poet['cut_parapraphs'] = filter_poet

print(poet.info())
poet.reset_index()
print(len(poet))
poet.to_csv('data/poet.csv', index=False)
