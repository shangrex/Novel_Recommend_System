'''
the script is to return the most similar poet from the input words
'''
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import *
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
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

# txt = "國破山河在，城春草木深。"
# txt = "舉杯邀明月，對影成三人。"
txt = "天若不愛酒，酒星不在天。"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer =  AutoTokenizer.from_pretrained('bert-base-chinese')

model = AutoModel.from_pretrained(model_name)
# print(model)
txt = tokenizer.tokenize(txt)

# print(txt)

seq_enc = tokenizer.convert_tokens_to_ids(txt)

input_tensor = torch.tensor([seq_enc])
# print("w2id", input_tensor)
# print(input_tensor.shape)
# input_tensor = pad_sequence(input_tensor, batch_first=True)
# print("padding", input_tensor)

mask_tensor = torch.zeros(input_tensor.shape, dtype=torch.long)
mask_tensor = mask_tensor.masked_fill(input_tensor != 0, 1)
# print("masking", mask_tensor)


output_seq = model(input_ids = input_tensor, attention_mask = mask_tensor)
# print(output_seq)
# print(len(output_seq))
# print(output_seq)
predict = output_seq[0]
print(predict.shape)
predict = torch.mean(predict, dim=1).squeeze()

print("output vector")
print(predict.shape)



compare = "花間一壺酒，獨酌無相親。"
compare = tokenizer.tokenize(compare)
compare = tokenizer.convert_tokens_to_ids(compare)
compare = torch.tensor([compare])
compare_mask = torch.zeros(compare.shape, dtype=torch.long)
compare_mask = compare_mask.masked_fill(compare != 0, 1)
compare = model(input_ids = compare, attention_mask = compare_mask)
compare = compare[0]
compare = torch.mean(compare, dim=1).squeeze()


print("predict shpae", predict.shape)
print("compare shape", compare.shape)



rst = cosine_similarity([predict.detach().numpy()], [compare.detach().numpy()])
print("result", rst)