import os
from dotenv import load_dotenv, find_dotenv
from ivus.data.dataset_managing import DataManager, SequenceReader

load_dotenv(find_dotenv())

CNN_PREDICTIONS = os.getenv('CNN_PREDICTIONS')
PROCESSED_DATA = os.getenv('PROCESSED_DATA')
MIN_TF_LENGTH = int(os.getenv('MIN_TF_LENGTH'))
MAX_TF_LENGTH = int(os.getenv('MAX_TF_LENGTH'))
STEP_TF_LENGTH = int(os.getenv('STEP_TF_LENGTH'))
CR_SIZE = int(os.getenv('CR_SIZE'))

dm = DataManager(processed_data=PROCESSED_DATA)
sr = SequenceReader(CNN_PREDICTIONS)


def timefold_generator(tf_length):
    sr.generate_timefolded_data(tf_length, train_rnn=True, val=True, test=True)


def shifted_generator():
    sr.generate_shifted_data(train_rnn=True)


def cropped_generator(window_size, stride=1):
    print('In cropped generator...')
    print(window_size)
    print(sr.generate_cropped_data)
    sr.generate_cropped_data(window_size, stride=stride, train_rnn=True)


def pullback_serializer():
    dm.save_all_pullbacks(
        train_cnn=True, val=True, train_rnn=True, test=True)
