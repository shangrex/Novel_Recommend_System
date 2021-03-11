from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence, text


class keras_gru():
    def __init__(self, vocab_size):
        # self.model = Sequential()
        # self.model.add(Embedding(len(vocab_size) + 1,
        #                     300,
        #                     weights=[embedding_matrix],
        #                     input_length=max_len,
        #                     trainable=False))
        # self.model.add(SpatialDropout1D(0.3))
        # self.model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
        # self.model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))

        # self.model.add(Dense(1024, activation='relu'))
        # self.model.add(Dropout(0.8))

        # self.model.add(Dense(1024, activation='relu'))
        # self.model.add(Dropout(0.8))

        # self.model.add(Dense(3))
        # self.model.add(Activation('softmax'))
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        pass
    
    def train(self, trianx, trainy):
        self.model.fit(trainx, trainy)

    def load(self, path):
        pass

    def save(self, path):
        pass