from src.model._base import basemodel
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

class bert(basemodel):
    def __init__(self):
        pass


    def forward(self, x):
        pass 


class transformers_bert():
    def __init__(self, model_name):
        self.tknzr = None
        self.model = None
    

    def enc(self, txt: str = 'bert-base-chinese'):
        maskpos = self.tknzr.encode(txt, add_special_tokens=True).index(103)
        print(maskpos)

    def save(self, path):
        self.tknzr.save_pretrained(path)
        self.model.save_pretrained(path)

    def download(self, model_name):
        self.tknzr = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def load(self, path):
        self.tknzr = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained()

    def predict(self, txt: str):
        inputs = self.tknzr(txt)
        outputs = self.model(**inputs)
        return outputs