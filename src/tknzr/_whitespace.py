from src.tknzr._base import basetknzr


class wstknzr(basetknzr):   
    def __init__(self):
        pass

    def tknz(self, txt: str): 
        return txt.split(' ')

    def dtknz(self, tkz):
        return " ".join(tkz)