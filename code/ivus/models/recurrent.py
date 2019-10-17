import os
from dotenv import load_dotenv, find_dotenv
from keras.layers import (
    Masking, Bidirectional, LSTM, TimeDistributed, Dense
)
from keras.initializers import glorot_uniform, orthogonal
from keras.optimizers import Adam
from .neural_nets import Recurrent
from .metrics import f1

load_dotenv(find_dotenv())

# Random Seed
random_seed = int(os.getenv('RANDOMSEED'))


def get_hrnn(beta_1, dropout, lr, lstm_units, tf_length=1):
    model_def = {
        'layers': [
            lambda: Masking(
                mask_value=-1, batch_input_shape=(None, None, tf_length)
            ),
            lambda: Bidirectional(LSTM(
                lstm_units, return_sequences=True, dropout=dropout,
                kernel_initializer=glorot_uniform(seed=random_seed),
                recurrent_initializer=orthogonal(seed=random_seed))),
            lambda: TimeDistributed(Dense(
                1, activation='sigmoid',
                kernel_initializer=glorot_uniform(seed=random_seed))),
        ],
        'optimizer': lambda: Adam(lr=lr, beta_1=beta_1),
        'metrics': ['accuracy', f1]
    }
    return Recurrent(**model_def)


def get_da_hrnn(params):
    print('params: {}'.format(params))
    layers = [
        lambda: Masking(
            mask_value=-1, batch_input_shape=(
                None, None, int(params.get('tf_length', 1)))
        )
    ]
    if int(params['recurrent_dropout']) == 0:
        layers.append(
            lambda: Bidirectional(LSTM(
                int(params['lstm_units']), return_sequences=True,
                kernel_initializer=glorot_uniform(seed=random_seed),
                recurrent_initializer=orthogonal(seed=random_seed))
            ),
        )
    else:
        layers.append(
            lambda: Bidirectional(LSTM(
                int(params['lstm_units']), return_sequences=True,
                recurrent_dropout=params['recurrent_dropout'],
                kernel_initializer=glorot_uniform(seed=random_seed),
                recurrent_initializer=orthogonal(seed=random_seed))
            )
        )
    layers.append(
        lambda: TimeDistributed(Dense(
            1, activation='sigmoid',
            kernel_initializer=glorot_uniform(seed=random_seed)))
    )
    model_def = {
        'layers': layers,
        'optimizer': lambda: Adam(lr=params['lr'], beta_1=params['beta_1']),
        'metrics': ['accuracy', f1]
    }
    return Recurrent(**model_def)
