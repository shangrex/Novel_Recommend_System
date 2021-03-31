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

model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
poem_dset = chpoemdset("test", tokenizer, limit_number)


BATCH_SIZE = 1
trainloader = DataLoader(poem_dset, batch_size=BATCH_SIZE, 
                         collate_fn=poem_dset.create_mini_batch
                         ,shuffle=True)

model.eval()
topk = 3

for m in range(20):
    print("test index: {}".format(m))
    dataiter = iter(trainloader).next()

    token_tensor , mask_tensor, lbl_tensor = dataiter[:3]

    outputs = model(input_ids = token_tensor,
                    attention_mask = mask_tensor)


    logits = outputs[0]
    # print("logits", type(logits))
    # print(logits.data)
    # _, pred = torch.max(logits.data, 1)
    predict_values, predict_indexes = torch.topk(logits.data, topk)
    predict_indexes = predict_indexes[0]
    for i, j in zip(token_tensor, lbl_tensor):
        # print(i)
        print(poem_dset.dtknz(i))
        for k in range(topk):
            index = predict_indexes[k].item()
            print("predict index", index)
            predict_author = torch.zeros(300)
            predict_author[index] = 1
            print("predict author", poem_dset.dauthor(predict_author))
        # print(j)
        print("ans", torch.argmax(j))
        print(poem_dset.dauthor(j))
                