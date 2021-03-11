from bs4 import BeautifulSoup
import pandas as pd

class XhtmlParser():
    def __init__(self):
        self.sentences = []
        self.novel_name= []
        self.novel_cls = []
        self.stop_words = ['{', '}']
        self.return_words = ['', ' ', '  ']
        
    def get_text(self, path):
        #open data
        data = open(path).read() 
        #bs4 parser
        soup = BeautifulSoup(data, 'html.parser')
        #find all parapraph        
        for i in soup.findAll('p'):
            check = True
            #check the werid paragraph
            for j in self.return_words:
                if i.text == j:
                    check = False

            #check the stop word 
            for j in self.stop_words:
                if j in i.text:
                    check = False

            if check == True:
                self.sentences.append(i.text)
        return self.sentences

    def get_name(self, novel_name):
        for i in self.sentences:
            self.novel_name.append(novel_name)


    def get_cls(self, novel_cls):
        for i in self.sentences:
            self.novel_cls.append(novel_cls)

    def get_data(self):
        submit = pd.DataFrame()
        submit['text'] = self.sentences
        submit['novel_name'] = self.novel_name
        submit['novel_cls'] = self.novel_cls
        return submit

