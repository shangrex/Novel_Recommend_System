from src.dset._base import basedset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from transformers import AdamW
from torch.nn import functional as F
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import torch 
import pysnooper
from torch.nn.utils.rnn import pad_sequence
from src.dset.util import trim_sequence


class chpoemdset(Dataset):
	def __init__(self, mode, tokenizer):
		self.data = pd.read_csv('data/poet.csv')
		
		self.stopwords = []
		fstp = open('data/stopwords/_ch_poem.txt', 'r')
		for i in fstp.read():
			self.stopwords.append(i)

		self.len = len(self.data)
		self.mode = mode
		self.tknzr = tokenizer
		self.c = Counter()
		self.c.update(self.data.author.values)
		self.lb2id = {}
		self.id2lb = {}
		self.author = []
		label_count = 0
		for lb, lb_count in self.c.most_common():
			self.id2lb[label_count] = lb
			self.lb2id[lb] = label_count
			self.author.append([lb])
			label_count += 1


		self.one_hot = OneHotEncoder()
		self.one_hot.fit(self.author)
		self.train_df, self.test_df = train_test_split(self.data,train_size=0.8, test_size=0.2, random_state=1)
		# self.train_df = self.data

	# @pysnooper.snoop()
	def __getitem__(self, idx: int):
		if self.mode == "test":
			txt = self.test_df['paragraphs'].iloc[idx]
			# print(self.test_df['author'].iloc[idx])
			lbl = self.one_hot.transform([[self.test_df['author'].iloc[idx]]])
			# lal = None
		else:
			txt = self.train_df['paragraphs'].iloc[idx]
			# print(txt)
			# print(self.train_df['author'].iloc[idx])
			lbl = self.one_hot.transform([[self.train_df['author'].iloc[idx]]])
		
		seq_enc = ['[CLS]']
		txt = self.tknzr.tokenize(txt)
		seq_enc += txt
		seq_enc = [i for i in seq_enc if i not in self.stopwords]


		seq_enc = self.tknzr.convert_tokens_to_ids(seq_enc)
		# print(seq_enc)
		# print(lbl.toarray())
		return (torch.tensor(seq_enc), torch.tensor(lbl.toarray()[0]))


	def __len__(self):
		if self.mode == "train":
			return len(self.train_df)
		else:
			self.mode == "test"
			return len(self.test_df)

	def dtknz(self, tkz):
		return self.tknzr.convert_ids_to_tokens(tkz)

	def len_lbl(self):
		return len(self.author)

	def create_mini_batch(self, samples):
		token_tensor = [s[0] for s in samples]
		lbl_tensor = torch.stack([s[1] for s in samples])
		token_tensor = pad_sequence(token_tensor, batch_first=True)
		token_tensor = trim_sequence(token_tensor, 200)
		# print("label tensor")
		# print(lbl_tensor)
		# print(type(lbl_tensor))
		# print("token tensor")
		# print(token_tensor)
		# print(type(token_tensor))


		mask_tensor = torch.zeros(token_tensor.shape, dtype=torch.long)
		mask_tensor = mask_tensor.masked_fill(token_tensor != 0, 1)
		return (token_tensor, mask_tensor, lbl_tensor)

	