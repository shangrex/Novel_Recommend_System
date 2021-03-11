def norm(tkz :list):
    '''
    clean the stopwords
    '''
    f = open('data/punctuation.txt', 'r', encoding = 'UTF-8')
    stopwords = []
    for i in f.read():
        stopwords.append(i)
    f.close()
    new_tkz = []
    for i in tkz:
        if i not in stopwords:
            new_tkz.append(i)
    return new_tkz