import numpy
import os
import ivus.experiments.state as state
import gc
import pickle
from ivus.data.dataset_managing import SequenceReader
from dotenv import load_dotenv, find_dotenv
from ivus.models.net_wrappers import HDARNN, NewNet
from time import time
from hyperopt import Trials, STATUS_OK, tpe, STATUS_FAIL, hp, fmin
# from keras.regularizers import l2
from keras import backend as K

load_dotenv(find_dotenv())

CNN_PREDICTIONS = os.getenv('CNN_PREDICTIONS')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER')

FOLDS = [3, 5, 9]
SAVE_BASE = os.path.join(RESULTS_FOLDER, os.path.basename(__file__))
MAX_EVALS = 100
EPOCHS = 200
BATCH_SIZE = 512
TRIALS = Trials()
eps = 10**(-3)

if not os.path.exists(SAVE_BASE):
    os.makedirs(SAVE_BASE)

space = {
    'beta_1': hp.uniform('beta_1', 0.7, 1-eps),
    'recurrent_dropout': hp.choice(
        'recurrent_dropout',
        (0, 0.1, 0.2, 0.3, 0.4)
    ),
    'lr': hp.loguniform('lr', numpy.log(0.0001), numpy.log(0.01)),
    'lstm_units': hp.quniform('lstm_units', 20, 60, 2),
}

sequence_reader = SequenceReader(CNN_PREDICTIONS, FOLDS)
csv_headers = [
    'model_id', 'fold', 'beta_1', 'lr', 'recurrent_dropout', 'lstm_units',
    'epoch', 'loss', 'acc', 'f1', 'val_loss', 'val_acc', 'val_f1', 'time']


def cross_validation(params, sequence_reader=sequence_reader):
    best_f1 = []

    for dataset in sequence_reader.get_rotated_dataset():
        current_fold = dataset['fold']

        print('-'*20+' ON FOLD {} '.format(current_fold)+'-'*20)

        print('X train shape: {}'.format(dataset['x_train_rnn'].shape))
        print('Y train shape: {}'.format(dataset['y_train_rnn'].shape))
        print('X validation shape: {}'.format(dataset['x_val'].shape))
        print('Y validation shape: {}'.format(dataset['y_val'].shape))

        print('Using: {}'.format(params))

        net = HDARNN(params, epochs=EPOCHS, batch_size=BATCH_SIZE)

        begin_time = time()

        result = net.train(
            dataset['x_train_rnn'],
            dataset['y_train_rnn'],
            x_val=dataset['x_val'],
            y_val=dataset['y_val'],
            shuffle=True,
        )

        elapsed_time = time() - begin_time

        max_f1 = numpy.max(result.history['val_f1'][50:])
        best_f1.append(max_f1)

        best_epoch = result.history['val_f1'].index(max_f1)

        if not state.created_csv:
            with open('{}/summary.csv'.format(SAVE_BASE), 'w') as csv_file:
                csv_file.write(', '.join(csv_headers))
                csv_file.write('\n')
            state.created_csv = True

        print('elapsed_time: {} {}'.format(elapsed_time, type(elapsed_time)))

        with open('{}/summary.csv'.format(SAVE_BASE), 'a') as csv_file:
            csv_file.write(', '.join(map(str, [
                state.model_id,
                current_fold,
                params['beta_1'],
                params['lr'],
                params['recurrent_dropout'],
                params['lstm_units'],
                best_epoch+1,
                result.history['loss'][best_epoch],
                result.history['acc'][best_epoch],
                result.history['f1'][best_epoch],
                result.history['val_loss'][best_epoch],
                result.history['val_acc'][best_epoch],
                max_f1,
                elapsed_time
            ])))
            csv_file.write('\n')
    return best_f1


def objective_function(params):
    K.clear_session()
    gc.collect()

    if len(TRIALS.trials) > 0:
        for trial in TRIALS.trials[:-1]:
            trial_hparams = dict([
                (key, value[0])
                for key, value in trial['misc']['vals'].items()
                if len(value) > 0
            ])
            if params == trial_hparams:
                print('MODEL ALREADY TESTED')
                print(params)
                print('Best validation f1: {}'.format(trial['result']['loss']))
                return {
                    'loss': trial['result']['loss'],
                    'status': STATUS_FAIL,
                    'model': trial['result']['model'],
                }

    f1 = cross_validation(params)

    validation_f1 = numpy.mean(f1)

    if len(state.top_5_best) < 5:
        print('Added model to early top 5')
        state.top_5_best.append(
            {
                'model_id': state.model_id,
                'val_f1': validation_f1,
                'params': params
            }
        )
    else:
        min_item = min(
            state.top_5_best, key=lambda x: x['val_f1'])
        if min_item['val_f1'] < validation_f1:
            print('Added model to top 5')
            index = state.top_5_best.index(min_item)
            state.top_5_best[index] = {
                'model_id': state.model_id,
                'val_f1': validation_f1,
                'params': params
            }

    state.model_id += 1

    print('Mean validation f1: {}'.format(validation_f1))
    return {'loss': -validation_f1, 'params': params, 'status': STATUS_OK}


best_hparams = fmin(
    fn=objective_function,
    space=space,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=TRIALS)

print("Best performing model chosen hyper-parameters:")
print(best_hparams)

with open('{}/top_5_best.txt'.format(SAVE_BASE),
          'wb') as top_5_txt:
    for model in state.top_5_best:
        top_5_txt.write('model_id: {}\n'.format(model['model_id']))
        top_5_txt.write('val_f1: {}\n'.format(model['val_f1']))
        top_5_txt.write('params:\n')
        for key, value in model['params'].items():
            top_5_txt.write('  {}: {}\n'.format(key, str(value)))

with open('{}/top_5_best.pkl'.format(SAVE_BASE),
          'wb') as top_5:
    pickle.dump(state.top_5_best, top_5)

with open('{}/top_5_best.pkl'.format(SAVE_BASE), 'rb') as top_5_best_file:
    top_5_best = pickle.load(top_5_best_file)

for rnn_model in top_5_best:
    model_base_path = '{}/model_{}'.format(SAVE_BASE, rnn_model['model_id'])
    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)
    model_save_path = model_base_path + '/models'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model = NewNet(
        'rnn', 10, save_path=model_save_path, epochs=1000,
        params=rnn_model['params'])
    sequence_reader = SequenceReader(CNN_PREDICTIONS, range(1, 11))

    for dataset in sequence_reader.get_rotated_dataset():
        history = model.train(
            dataset['x_train_rnn'], dataset['y_train_rnn'],
            x_val=dataset['x_val'], y_val=dataset['y_val']
        )
        model.next()
