from src.dset._ch_poem import chpoemdset_gen_author
from transformers import *
from transformers import GPT2Tokenizer, GPT2Model
import torch


num_beams = 3



# print(model)
# print(gen_poem[0])
# print(gen_poem.dtknz(gen_poem[0][0]))
# print(gen_poem.dtknz(gen_poem[0][1]))

# outputs = model.generate(gen_poem[0][1], max_length=200)
# outputs = model.generate(encoded_input)
# print(outputs)
# print(gen_poem.dtknz(outputs))


tokenizer = AutoTokenizer.from_pretrained(vocab_file="data/pretrain/poem_pretrain/vocab.txt", pretrained_model_name_or_path="data/pretrain/poem_pretrain/")
model = AutoModelForCausalLM.from_pretrained("data/pretrain/poem_pretrain/")
print(model)
input_context = "你好"
# get tokens of words that should not be generated
# bad_words_ids = [tokenizer(bad_word).input_ids for bad_word in ["idiot", "stupid", "shut up"]]
# encode input context
input_ids = tokenizer(input_context, return_tensors="pt").input_ids
print(input_ids)

# generate sequences without allowing bad_words to be generated
outputs = model.generate(input_ids=input_ids, max_length=40, do_sample=True)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))



gen_poem = chpoemdset_gen_author(tokenizer, "李白")
print(gen_poem.dtknz(gen_poem[0][0][:10]))
print(torch.unsqueeze(gen_poem[0][0][:10], 0))
# print(gen_poem[0][0])
outputs = model.generate(torch.unsqueeze(gen_poem[0][0][:10], 0), max_length=200)
print(outputs)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print(gen_poem.dtknz(torch.tensor([100, 101, 102, 103])))