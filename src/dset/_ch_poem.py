from src.dset._base import basedset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from transformers import AdamW
from torch.nn import functional as F
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import torch 
import numpy as np
import pysnooper
from torch.nn.utils.rnn import pad_sequence
from src.dset.util import trim_sequence


class chpoemdset(Dataset):
	def __init__(self, mode, tokenizer, num_limit_author):
		#read data
		self.data = pd.read_csv('data/poet.csv')
		#read stopwords
		self.stopwords = []
		fstp = open('data/stopwords/_ch_poem.txt', 'r')
		for i in fstp.read():
			self.stopwords.append(i)
		#set data's len
		self.len = len(self.data)
		#set data mode
		self.mode = mode
		self.tknzr = tokenizer
		self.c = Counter()
		self.c.update(self.data.author.values)
		self.lb2id = {}
		self.id2lb = {}
		self.author = []
		label_count = 0
		#count the number and label
		# for lb, lb_count in self.c.most_common():
		# 	self.id2lb[label_count] = lb
		# 	self.lb2id[lb] = label_count
		# 	self.author.append([lb])
		# 	label_count += 1
		# print(len(self.author))

		#limit the author 
		wanted_author = []
		for i, j in self.c.most_common():
			if j < num_limit_author:
				break
			wanted_author.append(i)
		filter_author = []
		for i in self.data['author']:
			if i in wanted_author:
				filter_author.append(True)
			else:
				filter_author.append(False)
	
		self.data = self.data.loc[filter_author]
		#make one hot encoding for author
		self.author = [[i] for i in wanted_author]

		self.one_hot = OneHotEncoder()
		self.one_hot.fit(self.author)
		self.train_df, self.test_df = train_test_split(self.data,train_size=0.95, test_size=0.05, random_state=1)
		# self.train_df = self.data

	# @pysnooper.snoop()
	def __getitem__(self, idx: int):
		if self.mode == "test":
			txt = self.test_df['paragraphs'].iloc[idx]
			# print(self.test_df['author'].iloc[idx])
			lbl = self.one_hot.transform([[self.test_df['author'].iloc[idx]]])
			# lal = None
		elif self.mode == "train":
			txt = self.train_df['paragraphs'].iloc[idx]
			# print(txt)
			# print(self.train_df['author'].iloc[idx])
			lbl = self.one_hot.transform([[self.train_df['author'].iloc[idx]]])
		else:
			txt = self.data['paragraphs'].iloc[idx]
			lbl = self.one_hot.transform([[self.data['author'].iloc[idx]]])

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
		elif self.mode == "test":
			return len(self.test_df)
		else:
			return len(self.data)

	def dtknz(self, tkz):
		return self.tknzr.convert_ids_to_tokens(tkz)

	def dauthor(self, author):
		author = np.array(author)
		return self.one_hot.inverse_transform([author])[0]

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

	def check_author(self):
		return [i[0] for i in self.author]

		






class chpoemdset_tag(Dataset):
	def __init__(self, mode, tokenizer, num_limit_tags):
		#read data
		self.data = pd.read_csv('data/poet.csv')
		self.data = self.data.dropna(subset=["tags"])
		#read stopwords
		self.stopwords = []
		fstp = open('data/stopwords/_ch_poem.txt', 'r')
		for i in fstp.read():
			self.stopwords.append(i)
		#set data's len
		self.len = len(self.data)
		#set data mode
		self.mode = mode
		self.tknzr = tokenizer
		self.c = Counter()
		self.c.update(self.data.tags.values)
		self.lb2id = {}
		self.id2lb = {}
		self.tags = []
		label_count = 0
		#count the number and label
		# for lb, lb_count in self.c.most_common():
		# 	self.id2lb[label_count] = lb
		# 	self.lb2id[lb] = label_count
		# 	self.author.append([lb])
		# 	label_count += 1
		# print(len(self.author))

		#limit the author 
		wanted_tags = []
		for i, j in self.c.most_common():
			if j < num_limit_tags:
				break
			wanted_tags.append(i)
		filter_tags = []
		for i in self.data['tags']:
			if i in wanted_tags:
				filter_tags.append(True)
			else:
				filter_tags.append(False)
	
		self.data = self.data.loc[filter_tags]
		#make one hot encoding for author
		self.tags = [[i] for i in wanted_tags]

		self.one_hot = OneHotEncoder()
		self.one_hot.fit(self.tags)
		self.train_df, self.test_df = train_test_split(self.data,train_size=0.95, test_size=0.05, random_state=1)
		# self.train_df = self.data

	# @pysnooper.snoop()
	def __getitem__(self, idx: int):
		if self.mode == "test":
			txt = self.test_df['paragraphs'].iloc[idx]
			lbl = self.one_hot.transform([[self.test_df['tags'].iloc[idx]]])
			# lal = None
		elif self.mode == "train":
			txt = self.train_df['paragraphs'].iloc[idx]
			# print(txt)
			# print(self.train_df['author'].iloc[idx])
			lbl = self.one_hot.transform([[self.train_df['tags'].iloc[idx]]])
		else:
			txt = self.data['paragraphs'].iloc[idx]
			lbl = self.one_hot.transform([[self.data['tags'].iloc[idx]]])

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
		elif self.mode == "test":
			return len(self.test_df)
		else:
			return len(self.data)

	def dtknz(self, tkz):
		return self.tknzr.convert_ids_to_tokens(tkz)

	def dauthor(self, author):
		author = np.array(author)
		return self.one_hot.inverse_transform([author])[0]

	def len_lbl(self):
		return len(self.tags)

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

	
	def nerf_data(self, limit_number = 300):
		wanted_author = []
		for i, j in self.c.most_common():
			if j > limit_number:
				break
			wanted_author.append(i)
		# assetturnover = assetturnover.loc[assetturnover['INDFMT']==b'FS']#I want FS

		# if self.mode == "train":
		# 	self.train_df = self.train_df.loc[self.train_df.author in wanted_author]
		# if self.mode == "test":
		# 	self.test_df = self.

	def to_csv(self, path):
		return self.data.to_csv(path)







class chpoemdset_gen_author(Dataset):
	def __init__(self, tokenizer, author, limit_number):
		#read data
		self.data = pd.read_csv('data/poet.csv')
		#read stopwords
		self.stopwords = []
		fstp = open('data/stopwords/_ch_poem.txt', 'r')
		for i in fstp.read():
			self.stopwords.append(i)
		#set data's len
		self.len = len(self.data)
		self.tknzr = tokenizer
		self.limit_number = limit_number
		#choose wanted author
		filter_author = []
		for i in self.data['author']:
			if i == author:
				filter_author.append(True)
			else:
				filter_author.append(False)
	
		self.data = self.data.loc[filter_author]

	# @pysnooper.snoop()
	def __getitem__(self, idx: int):
		txt = self.data['paragraphs'].iloc[idx]

		txt = self.tknzr.tokenize(txt)
		seq_enc = []
		seq_enc += txt
		seq_enc = [i for i in seq_enc if i not in self.stopwords]


		seq_enc = self.tknzr.convert_tokens_to_ids(seq_enc)
	
		return (torch.tensor(seq_enc)[:-1], torch.tensor(seq_enc)[1:])


	def __len__(self):
		return len(self.data)

	def dtknz(self, tkz):
		return self.tknzr.convert_ids_to_tokens(tkz)


	def create_mini_batch(self, samples):
		train_tensor = [s[0] for s in samples]
		test_tensor = [s[1] for s in samples]
		train_tensor = pad_sequence(train_tensor, batch_first=True)
		train_tensor = trim_sequence(train_tensor, self.limit_number)
		test_tensor = pad_sequence(test_tensor, batch_first=True)
		test_tensor = trim_sequence(test_tensor, self.limit_number)


		mask_tensor = torch.zeros(token_tensor.shape, dtype=torch.long)
		mask_tensor = mask_tensor.masked_fill(token_tensor != 0, 1)
		return (token_tensor, mask_tensor, lbl_tensor)

	

		
print("reload ch poem")
