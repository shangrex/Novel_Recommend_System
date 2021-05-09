'''
the script is to find the unsupervised topic from the poet
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd 
import pickle
import argparse


parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--num_com', type=int, required=True,
                    help='number of component')
parser.add_argument('--exp_name', type=str, required=True,
                    help='experiment\'s name')
parser.add_argument('--max_df', type=float, required=False, default=1.0,
                    help='max portion of freqency for finding topic')
parser.add_argument('--min_df', type=int, required=False, default=1,
                    help='max number of freqency for finding topic')
parser.add_argument('--seed', type=int, required=False, default=42,
                    help='random of all seed')
parser.add_argument('--max_iter', type=int, required=False, default=40,
                    help='max iteration to train LDA')
parser.add_argument('--topk', type=int, required=False, default=10,
                    help='topk topic features to store')


args = parser.parse_args()
num_com = args.num_com
max_df = args.max_df
min_df = args.min_df
exp_name = args.exp_name
seed = args.seed
topk = args.topk


filter_poet = pd.read_csv('data/poet.csv')

cv = CountVectorizer(max_df = max_df, min_df = min_df)
dtm = cv.fit_transform(filter_poet['cut_parapraphs'])
print("total countvector dimension", len(cv.get_feature_names()))

LDA = LatentDirichletAllocation(n_components=num_com, random_state=seed, max_iter=args.max_iter)
LDA.fit(dtm)


with open(f'data/topic/{exp_name}topic.pkl', 'wb') as f:
    rst = {}
    topic_words = []
    for i,topic in enumerate(LDA.components_):
        print(f"TOP 10 WORDS PER TOPIC #{i}")
        print([cv.get_feature_names()[index] for index in topic.argsort()[-10:]])
        topic_words = [cv.get_feature_names()[index] for index in topic.argsort()[-topk:]]
        rst[i] = topic_words
    pickle.dump(rst, f)

#save the pretain lda model
with open(f'data/pretrain/{exp_name}lda.pkl', 'wb') as f:
    pickle.dump(LDA, f)