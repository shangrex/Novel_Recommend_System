'''
the script is to return the most similar poet from the input words
'''
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import *
import tensorflow as tf
import os
import argparse
import torch 


parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--model_name', type=str, required=False,
                    help='pretrain\'s model name or model path',
                    default='bert-base-chinese')

parser.add_argument('--txt', type=str, required=True, 
                    help='the input sentence')

args = parser.parse_args()
txt = args.txt
model_name = args.model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')

model = AutoModel.from_pretrained(model_name)
print(model)
txt = tokenizer.tokenize(txt)

print(txt)

seq_enc = tokenizer.convert_tokens_to_ids(txt)

input_tensor = torch.tensor([seq_enc])
print("w2id", input_tensor)
print(input_tensor.shape)
# input_tensor = pad_sequence(input_tensor, batch_first=True)
# print("padding", input_tensor)

mask_tensor = torch.zeros(input_tensor.shape, dtype=torch.long)
mask_tensor = mask_tensor.masked_fill(input_tensor != 0, 1)
print("masking", mask_tensor)


output_seq = model(input_ids = input_tensor, attention_mask = mask_tensor)

print("output vector")
print(output_seq[0].shape)

