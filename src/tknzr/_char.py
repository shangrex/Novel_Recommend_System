from src.tknzr._base import basetknzr
from src.utils.tknzr import norm

class chartknzr(basetknzr):
    def __init__(self):
        super().__init__()

    def tknz(self, txt: str):
        return list(norm(txt))
    
    def dtknz(self, tkz :list):
        return ' '.join(tkz)
