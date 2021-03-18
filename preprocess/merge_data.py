import pandas as pd 


data = pd.read_csv('data/poet0.csv')
d2 = pd.read_csv('data/poet150000.csv')
d3 = pd.read_csv('data/poet300000.csv')


data = data.append(d2)
data = data.append(d3)

data.to_csv('data/delete.csv')
print(data.describe())    