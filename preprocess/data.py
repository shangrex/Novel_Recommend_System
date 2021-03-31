#make the novel data
# from read_file import XhtmlParser
from tqdm import tqdm
import pandas as pd
import os


# #make battle_through_the_heaven csv (斗破蒼穹)
# all_file = os.listdir('data/fascinate/斗破蒼穹/')
# parser = XhtmlParser()

# for i in all_file:
#     #read the file
#     parser.get_text('data/fascinate/斗破蒼穹/'+i)

# #label the sentences name
# parser.get_name('battle_through_the_heavens')
# #label the sentences class
# parser.get_cls('fantasy')
# #make the pandas
# submit = parser.get_data()
# #transform pandas to csv
# submit.to_csv('data/battle_through_the_heavens.csv', index=False)


# #make JinYong csv(金庸全集)
# all_file = os.listdir('data/jinyong/all_novel/')
# parser = XhtmlParser()

# for i in all_file:
#     #read the file
#     parser.get_text('data/jinyong/all_novel/'+i)

# #label the sentences name
# parser.get_name('jinyong')
# #label the sentences class
# parser.get_cls('short')
# #make the pandas
# submit = parser.get_data()
# #transform pandas to csv
# submit.to_csv('data/jinyong.csv', index=False)


# #make luxun csv (魯迅全集)
# all_file = os.listdir('data/luxun/all_novel/')
# parser = XhtmlParser()

# for i in all_file:
#     #read the file
#     parser.get_text('data/luxun/all_novel/'+i)

# #label the sentences name
# parser.get_name('luxun')
# #label the senteces class
# parser.get_cls('wuxia')
# #make the pandas
# submit = parser.get_data()
# #transform pandas to csv
# submit.to_csv('data/luxun.csv', index=False)


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
for i in tqdm(range(57)):
    f = pd.read_json('data/json/poet.tang.{}.json'.format(i*1000))
    poet = poet.append(f)
#read tang poet's strains
for i in tqdm(range(57)):
    f = pd.read_json('data/strains/json/poet.tang.{}.json'.format(i*1000))
    strains = strains.append(f)

poet = pd.merge(poet, strains, on="id")


#count poet length
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

hot_data = pd.DataFrame()
#0~25400
for i in tqdm(range(255)):
    f = pd.read_json('data/rank/poet/poet.song.rank.{}.json'.format(i*1000))
    hot_data = hot_data.append(f)
#0~57
for i in tqdm(range(57)):
    f = pd.read_json('data/rank/poet/poet.tang.rank.{}.json'.format(i*1000))
    hot_data = hot_data.append(f)



poet['baidu'] = hot_data['baidu'].to_numpy()
poet['so360'] = hot_data['so360'].to_numpy()
poet['bing'] = hot_data['bing'].to_numpy()
poet['bing_en'] = hot_data['bing_en'].to_numpy()
poet['google'] = hot_data['google'].to_numpy()
poet['rank'] = (hot_data['baidu']+hot_data['so360']+hot_data['bing']+hot_data['bing_en']+hot_data['google']).to_numpy() 

poet.reset_index(drop=True)
print(len(poet))
poet.to_csv('data/poet.csv', index=False)
