import pandas as pd 

data = pd.read_csv('data/poet.csv')
split_len = 150000
for i in range(0, len(data), split_len):
    tmp = data[i:i+split_len]
    print(i)
    tmp.to_csv('data/poet{}.csv'.format(i))


