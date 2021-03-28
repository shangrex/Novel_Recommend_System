from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import typing 
from src.utils.tknzr import norm

class basetknzr():    

    def  __init__(self):
        self.c = Counter()
        self.max_id = 0
        self.tk2id = {}
        self.id2tk = {}

    def save(self):
        pass

    def load(self):
        pass
    
    def fit_tfidf(self, tkz: list):
        self.tfidf_transformer = TfidfVectorizer()
        self.tfidf_transformer.fit(tkz)

    def fit_cntvec(self, tkz: list):
        self.cntvec_transformer = CountVectorizer()
        self.cntvec_transformer.fit(tkz)
        
    def enc_tfidf(self, tkz: list):
        return self.tfidf_transformer.transform(tkz)

    def enc(self, tkz: list):
        self.enc_char = []
        for i in tkz:
            if i not in self.tk2id:
                self.enc_char.append("UNK")
            else: 
                self.enc_char.append(self.tk2id[i])

        return self.enc_char

    def enc_cntvec(self, tkz: list):
        return self.cntvec_transformer.transform(tkz)

    def dec(self, tkz: list):
        self.dec = []
        for i in tkz:
            if i == "UNK":
                self.dec.append("UNK")
            else:
                self.dec.append(self.id2tk[i])
        return " ".join(self.dec)

    def build_vocab(self, tkz :list, min_count=0):
        self.c.update(norm(tkz))

        for tk, tk_count in self.c.most_common():
            
            if tk_count < min_count:
                break
            if tk in self.tk2id:
                continue
            self.tk2id[tk] = self.max_id
            self.id2tk[self.max_id] = tk
            self.max_id += 1
            
        return self.tk2id
