import pandas as pd
from tqdm import tqdm
import pickle

poet = pd.read_csv('data/poet.csv')

f = open('data/topic/1000_10topic.pkl', 'rb')
top_dict = pickle.load(f)
for i in tqdm(range(len(poet))):
    match_count = 0
    count = 0
    for j, k in top_dict.items():
        for l in k:
            if l in poet['paragraphs'].iloc[i]:
                count += 1
        if match_count < count:
            match_count = count
            tmp = k
    if match_count == 0:
        poet['tags'].iloc[i] = []
        continue 
    if type(poet['tags'].iloc[i]) == float or type(poet['tags'].iloc[i]) == type(None):
        poet['tags'].iloc[i] = tmp
    elif type(poet['tags'].iloc[i]) == list:
        poet['tags'].iloc[i] += tmp
    elif type(poet['tags'].iloc[i]) == str:
        poet['tags'].iloc[i] = list(poet['tags'].iloc[i])+tmp
    else:
        print(type(poet['tags'].iloc[i]))
        print(poet['tags'].iloc[i])
        print("something wrong")
    

    



poet.reset_index()
print(len(poet))


poet.to_csv('data/poet.csv', index=False)
