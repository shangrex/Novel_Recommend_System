import pandas as pd 

data = pd.read_csv('data/poet.csv', index_col=False)
print("before data")
print(data.info())
print(data.describe())
split_len = 20000
for i in range(0, len(data), split_len):
    tmp = data[i:i+split_len]
    print(i)
    tmp.to_csv('data/poet{}.csv'.format(i), index=False)


