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

parser.add_argument('--topk', type=int, required=False, default=1, 
                    help="topk\'s accuracy")

args = parser.parse_args()
model_path = args.model_path
limit_number = args.limit_number
topk = args.topk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')

        
poem_dset = chpoemdset("test", tokenizer, limit_number)


BATCH_SIZE = 4
trainloader = DataLoader(poem_dset, batch_size=BATCH_SIZE, 
                         collate_fn=poem_dset.create_mini_batch)


model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=poem_dset.len_lbl())
# model = BertForMaskedLM.from_pretrained(model_path)
print("model")
def get_valid_prediction(model):
    acc = 0
    # count = 0
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(trainloader):
            
            data = [i.to(device) for i in data]

            token_tensor , mask_tensor, lbl_tensor = data[:3]

    
            outputs = model(input_ids = token_tensor,
                            attention_mask = mask_tensor)

            logits = outputs[0]

            for b in range(logits.data.shape[0]):
                predict_values, predict_indexes = torch.topk(logits.data[b], topk)
                target = torch.argmax(lbl_tensor[b])
                # print("target", target)
                for k in range(predict_indexes.shape[0]):
                    index = predict_indexes[k].item()
                    # print("top{}".format(k), index)
                    if index == target:
                        correct += 1

            total += BATCH_SIZE
   
            # count += 1 
            # if  count > 1:
            #     break
        print(correct, total)
        acc = correct / total
    return acc




print("device:", device)
model = model.to(device)
print("valid testing ....")
acc = get_valid_prediction(model)
print("classification acc:", acc)