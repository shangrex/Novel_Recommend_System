#make the novel data
from read_file import XhtmlParser
import os


#make battle_through_the_heaven csv (斗破蒼穹)
all_file = os.listdir('data/fascinate/斗破蒼穹/')
parser = XhtmlParser()

for i in all_file:
    #read the file
    parser.get_text('data/fascinate/斗破蒼穹/'+i)

#label the sentences name
parser.get_name('battle_through_the_heavens')
#label the sentences class
parser.get_cls('fantasy')
#make the pandas
submit = parser.get_data()
#transform pandas to csv
submit.to_csv('data/battle_through_the_heavens.csv', index=False)


#make JinYong csv(金庸全集)
all_file = os.listdir('data/jinyong/all_novel/')
parser = XhtmlParser()

for i in all_file:
    #read the file
    parser.get_text('data/jinyong/all_novel/'+i)

#label the sentences name
parser.get_name('jinyong')
#label the sentences class
parser.get_cls('short')
#make the pandas
submit = parser.get_data()
#transform pandas to csv
submit.to_csv('data/jinyong.csv', index=False)


#make luxun csv (魯迅全集)
all_file = os.listdir('data/luxun/all_novel/')
parser = XhtmlParser()

for i in all_file:
    #read the file
    parser.get_text('data/luxun/all_novel/'+i)

#label the sentences name
parser.get_name('luxun')
#label the senteces class
parser.get_cls('wuxia')
#make the pandas
submit = parser.get_data()
#transform pandas to csv
submit.to_csv('data/luxun.csv', index=False)


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


for i in tqdm(range(57)):
    f = pd.read_json('data/json/poet.tang.{}.json'.format(i*1000))
    poet = poet.append(f)
#read tang poet's strains
for i in tqdm(range(57)):
    f = pd.read_json('data/strains/json/poet.tang.{}.json'.format(i*1000))
    strains = strains.append(f)

poet = pd.merge(poet, strains, on="id")

poet.reset_index(drop=True)
print(len(poet))
poet.to_csv('data/poet.csv')
