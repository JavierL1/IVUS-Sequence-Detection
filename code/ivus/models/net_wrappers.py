import os
from .convolutional import get_cnn
from keras.callbacks import ModelCheckpoint
from .recurrent import get_hrnn, get_da_hrnn
from .callbacks import CustomStopper

MODELS_SAVE_PATH = '../models'


class ClassicNet(object):
    def __init__(
        self, net_type, folds, save_path=None, name='', batch_size=32,
        epochs=100, callbacks=None, **kwargs
    ):
        self.net_type = net_type
        self.name = name
        self.folds = folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.current_fold = 1
        self.params = kwargs.get('params', None)
        self.save_path = save_path
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.instance_net()

    def instance_net(self):
        if self.net_type == 'cnn':
            self.net = get_cnn(name=self.name)

        elif self.net_type == 'rnn':
            self.net = get_hrnn(
                self.params['beta_1'],
                self.params['dropout'],
                self.params['lr'],
                int(self.params['lstm_units'])
            )
            es = CustomStopper(
                monitor='val_f1', patience=200, verbose=1,
                mode='max')
            self.callbacks = [es]
            if self.save_path is not None:
                mcp = ModelCheckpoint(
                    '{}/fold_{}_best.model'.format(
                        self.save_path, self.current_fold),
                    monitor='val_f1', save_best_only=True, mode='max')
                self.callbacks.append(mcp)

        elif self.net_type == 'tf_rnn':
            self.net = get_hrnn(
                self.params['beta_1'],
                self.params['dropout'],
                self.params['lr'],
                int(self.params['lstm_units']),
                int(self.params['tf_length'])
            )
            es = CustomStopper(
                monitor='val_f1', patience=200, verbose=1,
                mode='max')
            self.callbacks = [es]
            if self.save_path is not None:
                mcp = ModelCheckpoint(
                    '{}/fold_{}_best.model'.format(
                        self.save_path, self.current_fold),
                    monitor='val_f1', save_best_only=True, mode='max')
                self.callbacks.append(mcp)

    def train(
        self, x_train, y_train, x_val=[], y_val=[], shuffle=False
    ):
        print('Training on fold {}'.format(self.current_fold))
        history = self.net.train(
            x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
            x_val=x_val, y_val=y_val, callbacks=self.callbacks, shuffle=shuffle
        )
        print('Model trained')
        if self.save_path is not None:
            model_save_path = ''
            if self.net_type == 'cnn':
                model_save_path = (
                    '{}/JAVIER_CALCIO_'
                    '{}_100Ep_Fold{}_Of_10.model').format(
                        self.save_path, self.name.upper(), self.current_fold)
            else:
                model_save_path = '{}/fold_{}.model'.format(
                        self.save_path, self.current_fold)
            self.net.save(model_save_path)
            print('Model saved on {}'.format(model_save_path))
        return history

    def next(self):
        self.current_fold += 1
        self.net.clear_session()
        del self.net
        if self.current_fold <= self.folds:
            self.instance_net()


class HSimpleRNN(object):
    def __init__(self, params, epochs=200, batch_size=32):
        self.beta_1 = params['beta_1']
        self.dropout = params['dropout']
        self.lr = params['lr']
        self.lstm_units = int(params['lstm_units'])
        self.epochs = epochs
        self.batch_size = batch_size

        self.net = get_hrnn(
            self.beta_1, self.dropout, self.lr, self.lstm_units)

    def train(
        self, x_train, y_train, x_val=[], y_val=[], shuffle=False
    ):
        return self.net.train(
            x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
            x_val=x_val, y_val=y_val, shuffle=shuffle
        )


class HDARNN(object):
    def __init__(self, params, epochs=200, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = get_da_hrnn(params)

    def train(
        self, x_train, y_train, x_val=[], y_val=[], shuffle=False
    ):
        return self.net.train(
            x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
            x_val=x_val, y_val=y_val, shuffle=shuffle
        )


class NewNet(object):
    def __init__(
        self, net_type, folds, save_path=None, name='', batch_size=32,
        epochs=100, callbacks=None, **kwargs
    ):
        self.net_type = net_type
        self.name = name
        self.folds = folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.current_fold = 1
        self.params = kwargs.get('params', None)
        self.save_path = save_path
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.instance_net()

    def instance_net(self):
        if self.net_type == 'cnn':
            self.net = get_cnn(name=self.name)

        elif self.net_type == 'rnn':
            self.net = get_da_hrnn(self.params)
            es = CustomStopper(
                monitor='val_f1', patience=200, verbose=1,
                mode='max')
            self.callbacks = [es]
            if self.save_path is not None:
                mcp = ModelCheckpoint(
                    '{}/fold_{}_best.model'.format(
                        self.save_path, self.current_fold),
                    monitor='val_f1', save_best_only=True, mode='max')
                self.callbacks.append(mcp)

        elif self.net_type == 'tf_rnn':
            self.net = get_hrnn(
                self.params['beta_1'],
                self.params['dropout'],
                self.params['lr'],
                int(self.params['lstm_units']),
                int(self.params['tf_length'])
            )
            es = CustomStopper(
                monitor='val_f1', patience=200, verbose=1,
                mode='max')
            self.callbacks = [es]
            if self.save_path is not None:
                mcp = ModelCheckpoint(
                    '{}/fold_{}_best.model'.format(
                        self.save_path, self.current_fold),
                    monitor='val_f1', save_best_only=True, mode='max')
                self.callbacks.append(mcp)

    def train(
        self, x_train, y_train, x_val=[], y_val=[], shuffle=False
    ):
        print('Training on fold {}'.format(self.current_fold))
        history = self.net.train(
            x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
            x_val=x_val, y_val=y_val, callbacks=self.callbacks, shuffle=shuffle
        )
        print('Model trained')
        return history

    def next(self):
        self.current_fold += 1
        self.net.clear_session()
        del self.net
        if self.current_fold <= self.folds:
            self.instance_net()


class HTimeFoldRNN(object):
    def __init__(self, params, epochs=200, batch_size=32):
        self.beta_1 = params['beta_1']
        self.dropout = params['dropout']
        self.lr = params['lr']
        self.lstm_units = int(params['lstm_units'])
        self.tf_length = int(params['tf_length'])
        self.epochs = epochs
        self.batch_size = batch_size

        self.net = get_hrnn(
            self.beta_1, self.dropout, self.lr, self.lstm_units,
            self.tf_length)

    def train(
        self, x_train, y_train, x_val=[], y_val=[]
    ):
        return self.net.train(
            x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
            x_val=x_val, y_val=y_val
        )
