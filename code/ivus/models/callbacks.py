from keras.callbacks import EarlyStopping


class CustomStopper(EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0,
                 mode='auto', start_epoch=50):
        super(CustomStopper, self).__init__(
                monitor=monitor, min_delta=min_delta, patience=patience,
                verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super(CustomStopper, self).on_epoch_end(epoch, logs)