from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

data = pd.DataFrame({"hello":[i for i in range(10)]})
train_data, test_data = train_test_split(data, train_size=0.8, test_size=0.2, random_state=1)
print("train")
print(train_data)
print("test")
print(test_data)

print(data.iloc[0])

c = Counter()
data = pd.read_csv('data/poet.csv')


# sprint(data.author.values[:10])
c.update(data.author.values) 
lb2id = {}
id2lb = {}
author = []
label_count = 0
for lb, lb_count in c.most_common():
    id2lb[label_count] = lb
    lb2id[lb] = label_count
    label_count += 1
    author.append([lb])
# author = [["佚名"], ["宋太祖"], ["京師語"]]
print(author)
one_hot = OneHotEncoder()
one_hot.fit(author)
print(one_hot.transform([["佚名"]]).toarray())
