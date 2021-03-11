from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import *
from torch.utils.tensorboard import SummaryWriter

# from src.dset._ch_poem import chpoemdset
import src.dset
# import src

import tensorflow as tf
import os
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-chinese"
tokenizer =  AutoTokenizer.from_pretrained(model_name)
        
poem_dset = src.dset.chpoemdset("train", tokenizer)
# x = poem_dset[0]
# print(x[1])
# print(len(poem_dset))
# print(poem_dset.dtknz(x[0]))

BATCH_SIZE = 4
trainloader = DataLoader(poem_dset, batch_size=BATCH_SIZE, 
                         collate_fn=poem_dset.create_mini_batch)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=poem_dset.len_lbl())
print("model")

def get_train(model, epoch, exp_name):
    writer = SummaryWriter('..data/exp/{}.pth'.format(exp_name))
    model.train()
    total_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for e in range(epoch):
        running_loss = 0
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
        
        print(running_loss)
        writer.add_image('loss', running_loss, e)
        total_loss += running_loss

    # torch.save(model.save_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__), 'data/exp/{}.pth'.format(exp_name))))
    writer.flush()
    writer.close()
    return total_loss / epoch


exp_name = "one"
print("trainning....")
model = model.to(device)
avg_loss = get_train(model, 1, "exp_name")
print("avg loss", avg_loss)




model.save_pretrained('data/pretrain/{}/model.pth'.format(model_name))
# config.save('data/exp/{}/config.json'.format(exp_name))
# print(os.path.dirname(os.path.dirname(__file__)))