from transformers import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dset._ch_poem import chpoemdset

import tensorflow as tf
import os
import torch 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-chinese"
tokenizer =  AutoTokenizer.from_pretrained(model_name)

        
poem_dset = chpoemdset("test", tokenizer)


BATCH_SIZE = 4
trainloader = DataLoader(poem_dset, batch_size=BATCH_SIZE, 
                         collate_fn=poem_dset.create_mini_batch)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=poem_dset.len_lbl())
print("model")

def get_valid_prediction(model):
    acc = 0
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(trainloader):
            
            data = [i.to(device) for i in data]

            token_tensor , mask_tensor, lbl_tensor = data[:3]
            # print(token_tensor.shape)
            # for i in token_tensor:
            #     print(poem_dset.dtknz(i))
            # print(mask_tensor.shape)
            # print(lbl_tensor)
            outputs = model(input_ids = token_tensor,
                            attention_mask = mask_tensor)

            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            lbl_tensor = torch.argmax(lbl_tensor)
            total += len(pred)
            correct += (lbl_tensor == pred).sum().item()
   

        print(correct, total)
        acc = correct / total
    return acc





print("device:", device)
model = model.to(device)
print("valid testing ....")
acc = get_valid_prediction(model)
print("classification acc:", acc)