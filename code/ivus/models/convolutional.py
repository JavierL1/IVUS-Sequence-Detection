import os
from dotenv import load_dotenv, find_dotenv
from keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,
    BatchNormalization
)
from keras.initializers import glorot_uniform
from .neural_nets import Convolutional

load_dotenv(find_dotenv())

# Random Seed
random_seed = int(os.getenv('RANDOMSEED'))

# Width and height of input frames
w, h = (128, 128)

# Base line net architectures
mauricio_nets = {
    'cnn1a': {
        'layers': [
            lambda: Conv2D(
                16, (3, 3), activation='relu', input_shape=(w, h, 1),
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Flatten(),
            lambda: Dense(
                1, kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: Activation('sigmoid')
        ],
    },
    'cnn1b': {
        'layers': [
            lambda: Conv2D(
                32, (3, 3), activation='relu', input_shape=(w, h, 1),
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Flatten(),
            lambda: Dense(
                1, kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: Activation('sigmoid')
        ],
    },
    'cnn2a': {
        'layers': [
            lambda: Conv2D(
                64, (3, 3), activation='relu', input_shape=(w, h, 1),
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Flatten(),
            lambda: Dense(
                1, kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: Activation('sigmoid')
        ],
    },
    'cnn3a': {
        'layers': [
            lambda: Conv2D(
                64, (3, 3), activation='relu', input_shape=(w, h, 1),
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Flatten(),
            lambda: Dense(
                1, kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: Activation('sigmoid')
        ],
    },
    'cnn3b': {
        'layers': [
            lambda: Conv2D(
                64, (3, 3), activation='relu', input_shape=(w, h, 1),
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                64, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Flatten(),
            lambda: Dense(
                1, kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: Activation('sigmoid')
        ],
    },
    'cnn4a': {
        'layers': [
            lambda: Conv2D(
                32, (3, 3), activation='relu', input_shape=(w, h, 1),
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                16, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                16, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Flatten(),
            lambda: Dense(
                1, kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: Activation('sigmoid')
        ],
    },
    'cnn4b': {
        'layers': [
            lambda: Conv2D(
                64, (3, 3), activation='relu', input_shape=(w, h, 1),
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                64, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.1, seed=random_seed),
            lambda: Flatten(),
            lambda: Dense(
                1, kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: Activation('sigmoid')
        ],
    },
    'cnn4c': {
        'layers': [
            lambda: Conv2D(
                64, (3, 3), activation='relu', input_shape=(w, h, 1),
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.4, seed=random_seed),
            lambda: Conv2D(
                64, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.4, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.4, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.4, seed=random_seed),
            lambda: Flatten(),
            lambda: Dense(
                1, kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: Activation('sigmoid')
        ],
    },
    'cnn5a': {
        'layers': [
            lambda: Conv2D(
                64, (3, 3), activation='relu', input_shape=(w, h, 1),
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.4, seed=random_seed),
            lambda: Conv2D(
                64, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.4, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.4, seed=random_seed),
            lambda: Conv2D(
                32, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.4, seed=random_seed),
            lambda: Conv2D(
                16, (3, 3), activation='relu',
                kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: BatchNormalization(),
            lambda: MaxPooling2D(pool_size=(2, 2)),
            lambda: Dropout(0.4, seed=random_seed),
            lambda: Flatten(),
            lambda: Dense(
                1, kernel_initializer=glorot_uniform(seed=random_seed)),
            lambda: Activation('sigmoid')
        ],
    },
}


def get_cnn(name='cnn5a'):
    return Convolutional(**mauricio_nets[name])
