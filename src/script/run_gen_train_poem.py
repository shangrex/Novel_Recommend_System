from src.dset._ch_poem import chpoemdset_gen_author
from transformers import *
from transformers import GPT2Tokenizer, GPT2Model
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForCausalLM.from_pretrained("ckiplab/gpt2-base-chinese")


epoch=1
BATCH_SIZE=4

trainloader = DataLoader(chpoemdset_gen_author, batch_size=BATCH_SIZE,
                        collate_fn=chpoemdset_gen_author.create_mini_batch)
model.train()
def get_train(model, epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for e in range(epoch):
