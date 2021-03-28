from src.tknzr._base import basetknzr
from src.utils.tknzr import norm
import jieba


class jiebatknzr(basetknzr):
    def __init__(self):
        super().__init__()

    def tknz(self, txt: str):
        tkz = jieba.lcut(txt)
        return norm(tkz)

    def dtknz(self, tkz):
        return " ".join(tkz)

