import os
import numpy as np
import random as rn
import tensorflow as tf
from dotenv import load_dotenv, find_dotenv
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam

load_dotenv(find_dotenv())
random_seed = int(os.getenv('RANDOMSEED'))


np.random.seed(random_seed)
rn.seed(random_seed)
tf.set_random_seed(random_seed)


# Base net with common methods and attributes
class BaseNet(object):
    def __init__(self, **kwargs):
        if 'layers' in kwargs and kwargs['layers']:
            self.net = Sequential()
            for layer in kwargs['layers']:
                self.net.add(layer())
            self.optimizer = kwargs.get('optimizer', lambda: Adam())
            self.loss = kwargs.get('loss', 'binary_crossentropy')
            self.metrics = kwargs.get('metrics', ['accuracy'])
            self.net.compile(
                loss=self.loss,
                optimizer=self.optimizer(),
                metrics=self.metrics
            )
            self.net.summary()
        else:
            raise Exception('No layers provided')

    def train(
        self, x_train, y_train, batch_size=32, epochs=100,
        x_val=[], y_val=[], callbacks=None, shuffle=False
    ):

        validation_data = (x_val, y_val) if (
            x_val.any() and y_val.any()) else None

        self.history = self.net.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=validation_data, callbacks=callbacks,
            verbose=2, shuffle=shuffle)

        return self.history

    def save(self, save_path):
        self.net.save(save_path)

    def predict(self, x):
        return self.net.predict(x)

    def clear_session(self):
        del self.net
        tf.reset_default_graph()
        K.clear_session()


class Convolutional(BaseNet):
    def __init__(self, **kwargs):
        super(Convolutional, self).__init__(**kwargs)

    def predict(self, x):
        out_net = self.net.predict(x)
        return np.array([val for sublist in out_net for val in sublist])


class Recurrent(BaseNet):
    def __init__(self, **kwargs):
        super(Recurrent, self).__init__(**kwargs)
