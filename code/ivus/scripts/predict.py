import os
import sys
import pickle
from ivus.models.predict import make_predictions
from ivus.models.metrics import f1
from ivus.data.dataset_managing import SequenceReader
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

CNN_PREDICTIONS = os.getenv('CNN_PREDICTIONS')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER')
SUBSETS = {
    subset: True for subset in ['train_cnn', 'val', 'train_rnn', 'test']}
EXPERIMENTS = [
    {
        'name': 'hp_cr',
        'generator': lambda: sr.get_dataset(**SUBSETS)
    },
    {
        'name': 'hp_po',
        'generator': lambda: sr.get_dataset(**SUBSETS)
    },
    {
        'name': 'hp_pr',
        'generator': lambda: sr.get_dataset(pad=2, **SUBSETS)
    },
    {
        'name': 'hp_ro',
        'generator': lambda: sr.get_dataset(**SUBSETS)
    },
    {
        'name': 'hp_sh',
        'generator': lambda: sr.get_dataset(**SUBSETS)
    },
    {
        'name': 'hp_tf',
        'generator': lambda tf_length: sr.get_folded_dataset(
            tf_length, **SUBSETS)
    }]

objects = {'f1': f1}

with open(
    os.path.join(RESULTS_FOLDER, 'hp_tf', 'top_5_best.pkl'), 'rb'
) as tf_best_file:
    tf_best = pickle.load(tf_best_file)

exp_index = int(sys.argv[1])
model_index = int(sys.argv[2])
pre = int(sys.argv[3]) == 1
sr = SequenceReader(CNN_PREDICTIONS)


def get_tf_length(model_name):
    model_id = int(model_name.split('_')[-1])
    for model in tf_best:
        if model['model_id'] == model_id:
            return int(model['params']['tf_length'])


exp = EXPERIMENTS[exp_index]
exp_folder = os.path.join(RESULTS_FOLDER, exp['name'])
models_names = [
    model_name for model_name in os.listdir(exp_folder)
    if model_name.startswith('model')]
model_name = models_names[model_index]
model_folder = os.path.join(exp_folder, model_name)
tf_length = None
if exp['name'] == 'hp_tf':
    tf_length = get_tf_length(model_name)
make_predictions(
    model_folder,
    objects,
    exp['generator'],
    tf_length=tf_length,
    pre=pre,
    **SUBSETS)
