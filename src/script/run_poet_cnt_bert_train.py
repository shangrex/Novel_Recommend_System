'''
Train transfer poem's paragraphs to sentence embedding though BERT
'''
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.dset.util import trim_sequence
from tqdm import tqdm
from transformers import *
import tensorflow as tf
import os
import argparse
import torch 
import pandas as pd


parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--model_name', type=str, required=False,
                    help='pretrain\'s model name or model path',
                    default='bert-base-chinese')

parser.add_argument('--input', type=str, required=False,
                    default='data/poet.csv',
                    help='the input sentence')
parser.add_argument('--output', type=str, required=False,
                    default='data/poet_bert_sum.csv',
                    help='the input sentence')



args = parser.parse_args()
data = args.input
model_name = args.model_name

poet = pd.read_csv(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')

model = AutoModel.from_pretrained(model_name)
# print(model)


# #sentence embeddings
# txt = tokenizer.tokenize(txt)
# seq_enc = tokenizer.convert_tokens_to_ids(txt)

# input_tensor = torch.tensor([seq_enc])

# mask_tensor = torch.zeros(input_tensor.shape, dtype=torch.long)
# mask_tensor = mask_tensor.masked_fill(input_tensor != 0, 1)


# output_seq = model(input_ids = input_tensor, attention_mask = mask_tensor)
# input_tensor = output_seq[0]
# print(input_tensor.shape)
# predict = torch.mean(input_tensor, dim=1).squeeze()



rst = {}

print("=="*7+"transfering to vector"+"=="*7)
for i in tqdm(range(len(poet))):
    txt = tokenizer.tokenize(poet['paragraphs'].iloc[i])

    seq_enc = tokenizer.convert_tokens_to_ids(txt)

    tmp_tensor = torch.tensor([seq_enc])
    tmp_tensor = trim_sequence(tmp_tensor, 200)

    mask_tensor = torch.zeros(tmp_tensor.shape, dtype=torch.long)
    mask_tensor = mask_tensor.masked_fill(tmp_tensor != 0, 1)


    output_seq = model(input_ids = tmp_tensor, attention_mask = mask_tensor)
    predict = output_seq[0]
    predict = torch.mean(predict, dim=1).squeeze()

    rst[poet['id'].iloc[i]] = predict

rst = pd.DataFrame(rst, index=False)
rst.save_csv(args.output)