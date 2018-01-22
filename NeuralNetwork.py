from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
#import cPickle

class NeuralNetwork:
    def __init__(self):
        self.model = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(256, init='lecun_uniform', input_shape=(6,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(128, init='lecun_uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(24, init='lecun_uniform'))
        self.model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        # RMSprop optimizer is usually a good choice for recurrent neural networks.
        rms = RMSprop()
        self.model.compile(loss='mse', optimizer=rms)
        return self.model

    # @staticmethod
    # def save_model(model):
    #     try:
    #         f = open('combination23.save', 'wb')
    #         cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #         f.close()
    #     except Exception as e:
    #         print(e)

    # def load_model(self):
    #     try:
    #         f = open('combination23.save', 'rb')
    #         self.model = cPickle.load(f)
    #         f.close()
    #         return self.model
    #     except Exception as e:
    #         print(e)

    def getModel(self):
        return self.model


