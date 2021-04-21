'''
Implement a search engine.
'''
import pandas as pd 
import argparse


parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--find', type=str, required=True,
                    help='the word for searching')

args = parser.parse_args()

target = args.find.split(' ')
print(target)


poet = pd.read_csv('data/poet.csv')

result = {'author':[], 'paragraphs':[]}

print("=="*7+"finding"+"=="*7)
for i in range(len(poet)):
    check_count = 0
    for t in target:
        if poet['author'].iloc[i] == t:
            check_count += 1
        if t in poet['paragraphs'].iloc[i]:
            check_count += 1    
    if check_count >= len(target):
        result['author'].append({poet['author'].iloc[i]:poet['paragraphs'].iloc[i]})
        result['paragraphs'].append({poet['author'].iloc[i]:poet['paragraphs'].iloc[i]})

print("=="*7+"result"+"=="*7)
for i, j in result.items():
    print("=="*7,"search type", i, "=="*7)
    print("searching result", len(j))
    for k in j:
        print(k)
    

