'''
the script is to return similar author from the selected poet
'''
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import os
import argparse
import torch 

# from src.dset import chpoemdset
from src.dset._ch_poem import chpoemdset 


parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--epoch', type=int, required=True,
                    help='training epoch')
parser.add_argument('--exp_name', type=str, required=True,
                    help='experiment\'s name')
parser.add_argument('--limit_number', type=int, required=True,
                    help='limit number for target dimension')
parser.add_argument('--model_name', type=str, required=False,
                    help='pretrain\'s model name or model path',
                    default='bert-base-chinese')
parser.add_argument('--txt', type=str,  required=True, 
                    default='the input sentenc to predcit')

args = parser.parse_args()
model_name = args.model_name
txt = args.txt

tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')

model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
poem_dset = chpoemdset("test", tokenizer, limit_number)


BATCH_SIZE = 1
trainloader = DataLoader(poem_dset, batch_size=BATCH_SIZE, 
                         collate_fn=poem_dset.create_mini_batch
                         ,shuffle=True)

model.eval()

for m in range(20):
    print("test index: {}".format(m))
    dataiter = iter(trainloader).next()

    token_tensor , mask_tensor, lbl_tensor = dataiter[:3]

    outputs = model(input_ids = token_tensor,
                    attention_mask = mask_tensor)


    logits = outputs[0]
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
                