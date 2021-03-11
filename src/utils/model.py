from sklearn import preprocessing

class one_hot():
    def __init__(self):
        self.enc = preprocessing.OneHotEncoder()

    def fit(self, target):
        self.enc.fit(target)

    def transform(self, target :list):
        return self.enc.transform(target).toarray()

