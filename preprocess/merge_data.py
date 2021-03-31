import pandas as pd 


data = pd.read_csv('data/poet0.csv', index_col=False)
d2 = pd.read_csv('data/poet150000.csv', index_col=False)
d3 = pd.read_csv('data/poet300000.csv', index_col=False)


data = data.append(d2, ignore_index=True)
data = data.append(d3, ignore_index=True)

data.to_csv('data/delete.csv', index=False)

print("merge data")
print(data.info())
print(data.describe())    
