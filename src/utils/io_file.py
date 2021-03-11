from html.parser import HTMLParser
from html.entities import name2codepoint

class MyHTMLParser(HTMLParser):



    def handle_starttag(self, tag, attrs):
        pass 
        # print("Start tag:", tag)
        # for attr in attrs:
            # print("     attr:", attr)

    def handle_endtag(self, tag):
        pass
        # print("End tag  :", tag)

    def handle_data(self, data):
        #check the word can used
        check = True
        
        stop_words = ['{', '}']
        return_words = ['', ' ', '  ']
        
        #check the stop word 
        for i in stop_words:
            if i in data:
                check = False
        
        # #check p length
        # if len(data) < 5:
        #     check = False
        
        #check the werid paragraph
        for i in return_words:
            if i == data:
                check = False    
        
        if check == True:
            print("Data   :", data, type(data), len(data))
            return data

    def handle_comment(self, data):
        pass
        # print("Comment  :", data)

    def handle_entityref(self, name):
        pass 
        # c = chr(name2codepoint[name])
        # print("Named ent:", c)

    def handle_charref(self, name):
        pass
        # if name.startswith('x'):
        #     c = chr(int(name[1:], 16))
        # else:
        #     c = chr(int(name))
        # print("Num ent  :", c)

    def handle_decl(self, data):
        pass
        # print("Decl     :", data)