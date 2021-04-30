'''
the script is to return similar author from the selected poet
'''
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import *
from torch.utils.tensorboard import SummaryWriter
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

args = parser.parse_args()
epoch = args.epoch
exp_name = args.exp_name
limit_number = args.limit_number
model_name = args.model_name



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')
   
poem_dset = chpoemdset("train", tokenizer, limit_number)


BATCH_SIZE = 4
trainloader = DataLoader(poem_dset, batch_size=BATCH_SIZE, 
                         collate_fn=poem_dset.create_mini_batch)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=poem_dset.len_lbl())
print(model)
print("model")

def get_train(model, epoch, exp_name):
    writer = SummaryWriter('data/exp/{}.pth'.format(exp_name))

    model.train()
    total_loss = 0
    count_x = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for e in range(epoch):
        running_loss = 0
        count_loss = 0
        for data in tqdm(trainloader):
            data = [i.to(device) for i in data]
            optimizer.zero_grad()

            token_tensor , mask_tensor, lbl_tensor = data[:3]
            # print(token_tensor.shape)
            # print(lbl_tensor)
            # print(lbl_tensor.shape)
            lbl_tensor = torch.argmax(lbl_tensor, 1)
            # print(lbl_tensor)
            outputs = model(input_ids = token_tensor,
                            attention_mask = mask_tensor,
                            labels = lbl_tensor)

            # print("optimizer start...")
            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            writer.add_scalar('loss', loss.item(), count_x)
            count_x += 1
            count_loss += 1
        print(running_loss/count_loss)
        writer.add_scalar(' avg loss', running_loss/count_loss, e)
        total_loss += running_loss/count_loss

    writer.flush()
    writer.close()
    return total_loss / epoch


print("trainning....")
model = model.to(device)
avg_loss = get_train(model, epoch, exp_name)
print("avg loss", avg_loss)




model.save_pretrained('data/pretrain/{}/'.format(exp_name))