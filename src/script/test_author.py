from transformers import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorflow as tf
import os
import torch 
import argparse

from src.dset._ch_poem import chpoemdset


parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--model_path', type=str, required=False,
                    help='experiment\'s name', default='bert-base-chinese')

parser.add_argument('--limit_number', type=int, required=True,
                    help='limit number for target dimension')


args = parser.parse_args()
model_path = args.model_path
limit_number = args.limit_number


tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')

model = BertForMaskedLM.from_pretrained(model_path)
        
poem_dset = chpoemdset("test", tokenizer, limit_number)


BATCH_SIZE = 1
trainloader = DataLoader(poem_dset, batch_size=BATCH_SIZE, 
                         collate_fn=poem_dset.create_mini_batch)

model.eval()

dataiter = iter(trainloader)

token_tensor , mask_tensor, lbl_tensor = dataiter[:3]

outputs = model(input_ids = token_tensor,
                attention_mask = mask_tensor)


logits = outputs[0]
_, pred = torch.max(logits.data, 1)
for i, j, k in zip(token_tensor, lbl_tensor, pred):
    print(i)
    print(poem_dset.dtknz(i))
    print(k)
    print("predict", torch.argmax(k))
    # k = torch.zeros(300)
    # k[torch.argmax(k)] = 1
    # print(poem_dset.dauthor(k))
    print(j)
    print("ans", torch.argmax(j))
    print(poem_dset.dauthor(j))
    print(j.shape)   
            