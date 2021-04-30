'''
Transfer poem's paragraphs to sentence embedding
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

parser.add_argument('--data', type=str, required=False,
                    default='data/poet.csv',
                    help='the input sentence')


args = parser.parse_args()
data = args.data
model_name = args.model_name

poet = pd.read_csv(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')

model = AutoModel.from_pretrained(model_name)
# print(model)

rst = {}

print("=="*7+"transfering to vector"+"=="*7)
for i in tqdm(range(len(poet))):
    txt = tokenizer.tokenize(i['paragraphs'])

    # print(txt)

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


pd.save_csv('data/poet_bert.csv')