import numpy as np
import os
import pickle
from dotenv import load_dotenv, find_dotenv
from .preprocessing import CSVReader, get_labels
from .postprocessing import (
    read_data_pullbacks, time_folder_fourier, vector_padding, shifts, padding,
    cropping)
from sklearn.model_selection import KFold
from keras.models import load_model
from tqdm import tqdm

load_dotenv(find_dotenv())

# Random Seed
RANDOM_SEED = int(os.getenv('RANDOMSEED'))
FOLDED_DATA = os.getenv('FOLDED_DATA')
SHIFTED_DATA = os.getenv('SHIFTED_DATA')
CROPPED_DATA = os.getenv('CROPPED_DATA')
ROTATED_DATA = os.getenv('ROTATED_DATA')
PULLBACKS_DATA = os.getenv('PULLBACKS_DATA')


class Pullback(object):
    def __init__(self, x, y, name, index, length, label_1=[], label_2=[]):
        self.x = x
        self.y = y
        self.label_1 = label_1
        self.label_2 = label_2
        self.name = name
        self.index = index
        self.length = length
        self.subsets = {}

    def __str__(self):
        return '{}\n{}'.format(self.name, self.subsets)

    def set_subset_in_fold(self, fold, subset):
        self.subsets[fold] = subset

    def get_subset_in_fold(self, fold):
        return self.subsets[fold]


def get_pullbacks_fold(indices_pullbacks, **kwargs):
    kf1 = KFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
    kf2 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    kf3 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    current_fold = 1

    for split_a, split_b in kf1.split(indices_pullbacks):
        for (split_a1, split_a2), (split_b1, split_b2) in zip(
            kf2.split(split_a), kf3.split(split_b)
        ):
            result = {}

            for key, value in kwargs.items():
                if value:
                    if key == 'train_cnn':
                        result[key] = split_a[split_a1]
                    elif key == 'val':
                        result[key] = split_a[split_a2]
                    elif key == 'train_rnn':
                        result[key] = split_b[split_b1]
                    elif key == 'test':
                        result[key] = split_b[split_b2]

            result['fold'] = current_fold

            yield result if result else None

            current_fold += 1


class DataManager(object):

    # CSV reader
    reader = CSVReader()
    n_pullbacks = 0
    indices_pullbacks = []

    def __init__(self, processed_data=None):
        # Data paths
        self.name_pullbacks_x = '{}/PULLBACKS_X_COUGAR.csv'.format(
            processed_data)
        self.name_pullbacks_y = '{}/PULLBACKS_Y_COUGAR.csv'.format(
            processed_data)
        self.name_pullbacks_names = '{}/PULLBACKS_NAMES_COUGAR.csv'.format(
            processed_data)

        # Read raw data from csv
        (
            self.X, self.Y, self.start_indices, self.end_indices,
            self.pullback_names
        ) = self.reader.read_pullbacks_from_CSV(
            names=self.name_pullbacks_names,
            file_x=self.name_pullbacks_x,
            file_y=self.name_pullbacks_y,
            return_names=True)

        self.unique_pullback_names = np.unique(self.pullback_names)

        # Normalize pixels
        self.X = self.X / 255.0
        # Get calcio
        self.Y = self.Y[:, 2]
        # Get amount of pullbacks
        self.n_pullbacks = len(self.start_indices)
        # Pullbacks iterator
        self.indices_pullbacks = range(self.n_pullbacks)

    def get_dataset(self, **kwargs):
        for pullbacks in get_pullbacks_fold(self.indices_pullbacks, **kwargs):
            result = {}
            result['fold'] = pullbacks['fold']
            for subset in [
                subset for subset, value in kwargs.items() if value
            ]:
                subset_frames = None
                # For every pullback in the subset
                for pullback in pullbacks[subset]:
                    print(pullback)
                    # Gets indices of frames inside current pullback
                    new_indices = [
                        int(i) for i in range(
                            self.start_indices[pullback],
                            self.end_indices[pullback] + 1)
                    ]
                    # Appends new frames indices into data subset frames
                    if subset_frames is None:
                        subset_frames = new_indices
                    else:
                        subset_frames = np.r_[
                            subset_frames, new_indices]
                # Add subset of frames to result
                result['x_'+subset] = self.X[subset_frames, :]
                result['y_'+subset] = self.Y[subset_frames]
                result['names_'+subset] = [
                    self.pullback_names[i] for i in subset_frames]
                print('Subset {} generated'.format(subset))

            yield result if result else None

    def predict_pullbacks_cnn(
        self, models_path, model_name, save_dir, **kwargs
    ):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for pullbacks in get_pullbacks_fold(self.indices_pullbacks, **kwargs):
            '''
            model_path = (
                '{}/JAVIER_CALCIO_'
                '{}_100Ep_Fold{}_Of_10.model'
            ).format(models_path, model_name.upper(), pullbacks['fold'])
            '''
            model_path = (
                '{}/fold_{}.model'
            ).format(models_path, pullbacks['fold'])
            model = load_model(model_path)

            fold_dir = save_dir + '/' + str(pullbacks['fold']) + '/CSV'
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            for subset in [
                subset for subset, value in kwargs.items() if value
            ]:
                print('Predicting for {} subset on fold {}...'.format(
                    subset, pullbacks['fold']))

                subset_dir = fold_dir + '/' + subset.upper()
                if not os.path.exists(subset_dir):
                    os.makedirs(subset_dir)
                # For every pullback in the subset
                for pullback in pullbacks[subset]:
                    # Gets indices of frames inside current pullback
                    frame_indices = [
                        int(i) for i in range(
                            self.start_indices[pullback],
                            self.end_indices[pullback] + 1)
                    ]
                    # Gets name, x, y and prediction of current pullback
                    name_pullback = self.unique_pullback_names[pullback]
                    x_pullback = self.X[frame_indices, :]
                    y_predicted = np.asarray(model.predict(x_pullback))
                    y_pullback = self.Y[frame_indices]
                    # Sets save path and writes file
                    save_path = subset_dir + '/' + name_pullback + '.csv'
                    with open(save_path, 'w') as pb_file:
                        for (y_hat, y_real) in zip(
                            y_predicted, y_pullback
                        ):
                            pb_file.write('{}, {}\n'.format(
                                y_hat[0], y_real
                            ))
                    print('Prediction saved on {}'.format(save_path))
                print('Subset predicted')

    def get_pullbacks(self, pullbacks):
        for pullback in pullbacks:
            # Gets indices of frames inside current pullback
            frame_indices = [
                int(i) for i in range(
                    self.start_indices[pullback],
                    self.end_indices[pullback] + 1)
            ]
            # Gets name, x, y and prediction of current pullback
            name_pullback = self.unique_pullback_names[pullback]
            x_pullback = self.X[frame_indices, :]
            y_pullback = self.Y[frame_indices]

            yield {
                'x': x_pullback,
                'y': y_pullback,
                'name': name_pullback,
            }

    def get_all_pullbacks(self, **kwargs):
        labels_1 = get_labels(1)
        labels_2 = get_labels(2)
        for index in range(self.n_pullbacks):
            frame_indices = [
                int(i) for i in range(
                    self.start_indices[index],
                    self.end_indices[index] + 1)
            ]

            pb_name = self.unique_pullback_names[index]

            pullback = Pullback(
                self.X[frame_indices, :],
                self.Y[frame_indices],
                pb_name,
                index,
                len(frame_indices),
                labels_1=labels_1[pb_name],
                labels_2=labels_2[pb_name]
            )

            for fold_pullbacks in get_pullbacks_fold(
                self.indices_pullbacks, **kwargs
            ):
                for subset in [
                    subset for subset, value in kwargs.items() if value
                ]:
                    if pullback.index in fold_pullbacks[subset]:
                        pullback.set_subset_in_fold(
                            fold_pullbacks['fold'], subset)
            yield pullback

    def save_all_pullbacks(self, **kwargs):

        if not os.path.exists(PULLBACKS_DATA):
            os.makedirs(PULLBACKS_DATA)

        labels_1 = get_labels(1)
        labels_2 = get_labels(2)

        for index in tqdm(range(self.n_pullbacks)):
            frame_indices = [
                int(i) for i in range(
                    self.start_indices[index],
                    self.end_indices[index] + 1)
            ]

            pb_name = self.unique_pullback_names[index]

            pullback = Pullback(
                self.X[frame_indices, :], self.Y[frame_indices],
                pb_name, index,
                len(frame_indices),
                label_1=labels_1[pb_name],
                label_2=labels_2[pb_name]
            )

            for fold_pullbacks in get_pullbacks_fold(
                self.indices_pullbacks, **kwargs
            ):
                for subset in [
                    subset for subset, value in kwargs.items() if value
                ]:
                    if pullback.index in fold_pullbacks[subset]:
                        pullback.set_subset_in_fold(
                            fold_pullbacks['fold'], subset)

            with open(
                '{}/{}.pkl'.format(PULLBACKS_DATA, pullback.name), 'wb'
            ) as pullback_file:
                pickle.dump(pullback, pullback_file)


class SequenceReader(object):
    def __init__(self, signals_folder, folds=range(1, 11)):
        self.signals_folder = signals_folder
        self.folds = folds

    def get_dataset(self, pad=1, labels='union', folds=None, **kwargs):
        if folds is not None:
            self.folds = folds

        for fold in self.folds:

            data_subsets = {}

            for key, value in kwargs.items():
                if value:
                    data_subsets[key] = read_data_pullbacks(
                        '{}/{}/CSV/{}/'.format(
                            self.signals_folder, fold, key.upper()),
                        pad=pad
                    )

            result = {}

            for data_subset, info in data_subsets.items():
                result['x_'+data_subset] = np.reshape(
                    info['x'],
                    (info['n_sequences'], info['max_len'], 1),
                )
                if labels == 'union':
                    result['y_'+data_subset] = np.reshape(
                        info['y'],
                        (info['n_sequences'], info['max_len'], 1),
                    )
                elif labels == 'inter':
                    result['y_'+data_subset] = []
                    for pb_name in info['names']:
                        with open(
                            os.path.join(PULLBACKS_DATA, pb_name+'.pkl'),
                            'rb'
                        ) as pb_file:
                            pb_data = pickle.load(pb_file)
                        intersection = pb_data.label_1 * pb_data.label_2
                        result['y_'+data_subset].append(
                            np.pad(
                                intersection,
                                (0, info['max_len']-len(intersection)),
                                'constant',
                                constant_values=-1)
                        )
                    result['y_'+data_subset] = np.stack(
                        result['y_'+data_subset]
                    )
                    result['y_'+data_subset] = np.reshape(
                        result['y_'+data_subset],
                        (info['n_sequences'], info['max_len'], 1),
                    )
                elif labels == 'mean':
                    result['y_'+data_subset] = []
                    for pb_name in info['names']:
                        with open(
                            os.path.join(PULLBACKS_DATA, pb_name+'.pkl'),
                            'rb'
                        ) as pb_file:
                            pb_data = pickle.load(pb_file)
                        mean = (pb_data.label_1 + pb_data.label_2) / 2
                        result['y_'+data_subset].append(
                            np.pad(
                                mean,
                                (0, info['max_len']-len(mean)),
                                'constant',
                                constant_values=-1)
                        )
                    result['y_'+data_subset] = np.stack(
                        result['y_'+data_subset]
                    )
                    result['y_'+data_subset] = np.reshape(
                        result['y_'+data_subset],
                        (info['n_sequences'], info['max_len'], 1),
                    )
                result['names_'+data_subset] = info['names']

            result['fold'] = fold

            yield result if result else None

    def generate_timefolded_data(
        self, timefold_length, folds=None, verbose=False, **kwargs
    ):
        if folds is not None:
            self.folds = folds

        save_folder = (FOLDED_DATA + '/length_'+str(timefold_length))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        print('Saving timefolded data in {}'.format(save_folder))

        for fold in self.folds:
            data_subsets = {}
            lengths = []

            for key, value in kwargs.items():
                if value:
                    data_subsets[key] = read_data_pullbacks(
                        '{}/{}/CSV/{}/'.format(
                            self.signals_folder, fold, key.upper()),
                        pad=False,
                    )

            for data_subset, info in data_subsets.items():

                x = time_folder_fourier(info['x'], timefold_length)
                x, y = vector_padding(
                    x, info['y'], info['max_len'], timefold_length)

                folded_data = {
                    'x': x,
                    'y': y,
                    'names': info['names']
                }

                save_file = (
                    save_folder + '/fold_' + str(fold) + '_' + data_subset
                )
                with open(save_file, 'wb') as output:
                    pickle.dump(folded_data, output)

                lengths += info['lengths']

            if verbose and fold == 1:
                print('max len: {}'.format(np.max(lengths)))
                print('mean len: {}'.format(np.mean(lengths)))
                print('std len: {}'.format(np.std(lengths)))
                print('median len: {}'.format(np.median(lengths)))
                print('min len: {}'.format(np.min(lengths)))

    def get_folded_dataset(self, timefold_length, folds=None, **kwargs):
        if folds is not None:
            self.folds = folds

        load_folder = (FOLDED_DATA + '/length_'+str(timefold_length))

        for fold in self.folds:
            result = {}
            result['fold'] = fold
            load_fold_path = load_folder+'/fold_'+str(fold)+'_'

            for key, value in kwargs.items():
                if value:
                    load_path = load_fold_path + key
                    with open(load_path, 'rb') as data_file:
                        data = pickle.load(data_file)
                        result['x_'+key] = data['x']
                        result['y_'+key] = data['y']
                        #result['names_'+key] = data['names']

            yield result if result else None

    def generate_shifted_data(
        self, folds=None, save_data=True, **kwargs
    ):
        if folds is not None:
            self.folds = folds

        if not os.path.exists(SHIFTED_DATA):
            os.makedirs(SHIFTED_DATA)

        print('Saving shifted data on {}'.format(SHIFTED_DATA))

        for fold in self.folds:
            data_subsets = {}

            for key, value in kwargs.items():
                if value:
                    data_subsets[key] = read_data_pullbacks(
                        '{}/{}/CSV/{}/'.format(
                            self.signals_folder, fold, key.upper()),
                        pad=False,
                    )

            for data_subset, info in data_subsets.items():

                x, y, max_len = shifts(info['x'], info['y'])
                x = padding(x, max_len)
                y = padding(y, max_len)

                print('Fold {}'.format(fold))
                print('Shape x_{}: {}'.format(data_subset, x.shape))
                print('Shape y_{}: {}'.format(data_subset, y.shape))

                folded_data = {
                    'x': x,
                    'y': y,
                }

                if save_data:
                    save_file = (
                        SHIFTED_DATA + '/fold_' + str(fold) + '_' + data_subset
                    )
                    with open(save_file, 'wb') as output:
                        pickle.dump(folded_data, output)

    def get_shifted_dataset(self, folds=None):
        if folds is not None:
            self.folds = folds

        subset = 'train_rnn'

        for result in self.get_dataset(val=True):
            load_fold_path = SHIFTED_DATA+'/fold_'+str(result['fold'])+'_'
            load_path = load_fold_path + subset
            with open(load_path, 'rb') as data_file:
                data = pickle.load(data_file)
                result['x_'+subset] = data['x']
                result['y_'+subset] = data['y']

            yield result if result else None

    def generate_cropped_data(
        self, window_size, stride=1, folds=None, save_data=True, **kwargs
    ):
        if folds is not None:
            self.folds = folds

        save_folder = CROPPED_DATA+'/size_'+str(window_size)
        print('Saving cropped data on {}'.format(save_folder))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for fold in self.folds:
            data_subsets = {}
            result = {}
            for key, value in kwargs.items():
                if value:
                    data_subsets[key] = read_data_pullbacks(
                        '{}/{}/CSV/{}/'.format(
                            self.signals_folder, fold, key.upper()),
                        pad=False,
                    )

            for data_subset, info in data_subsets.items():

                x, meta_x = cropping(info['x'], window_size)
                y, meta_y = cropping(info['y'], window_size)

                #print('Fold {}'.format(fold))
                #print('Shape x_{}: {}'.format(data_subset, x.shape))
                #print('Shape y_{}: {}'.format(data_subset, y.shape))

                result['x_'+data_subset] = {
                    'data': x,
                    'meta': meta_x
                }
                result['y_'+data_subset] = {
                    'data': y,
                    'meta': meta_y
                }

                if save_data:
                    folded_data = {
                        'x': x,
                        'y': y
                    }
                    save_file = (
                        save_folder + '/fold_' + str(fold) + '_' + data_subset
                    )
                    with open(save_file, 'wb') as output:
                        pickle.dump(folded_data, output)
                
            yield result

    def get_cropped_dataset(self, window_size, folds=None):
        if folds is not None:
            self.folds = folds

        load_folder = CROPPED_DATA+'/size_'+str(window_size)
        subset = 'train_rnn'

        for result in self.get_dataset(val=True):
            load_path = (
                load_folder + '/fold_' + str(result['fold']) + '_' + subset)
            with open(load_path, 'rb') as data_file:
                data = pickle.load(data_file)
                result['x_'+subset] = data['x']
                result['y_'+subset] = data['y']
            yield result if result else None

    def get_rotated_dataset(self, folds=None):
        if folds is not None:
            self.folds = folds

        subset = 'train_rnn'

        print('Saving rotated data on {}'.format(ROTATED_DATA))

        for result in self.get_dataset(val=True):
            load_fold_path = ROTATED_DATA+'/fold_'+str(result['fold'])
            load_subset_path = load_fold_path + '/' + subset
            filenames = os.listdir(load_subset_path)

            pullback_files = [
                filename for filename in filenames
                if filename.endswith('.csv')
            ]

            raw_data_y = []
            lengths = []

            for pullback_file in pullback_files:
                pullback_name = pullback_file.split('/')[-1]\
                                    .split('.')[0]
                with open(
                    PULLBACKS_DATA+'/'+pullback_name+'.pkl'
                ) as pb_data:
                    pullback = pickle.load(pb_data)

                raw_data_y.append(pullback.y)
                lengths.append(pullback.length)

            max_length = np.max(lengths)
            data_x = []

            for pullback_file in pullback_files:
                with open(
                    load_subset_path+'/'+pullback_file, 'rb'
                ) as x_pb_data:
                    x_pullback = np.loadtxt(x_pb_data)
                n_rotations = x_pullback.shape[0]
                x_pullback = np.pad(
                    x_pullback,
                    ((0, 0), (0, max_length-x_pullback.shape[1])),
                    'constant', constant_values=-1
                )
                data_x.append(x_pullback)

            data_y = []
            for y in raw_data_y:
                data_y += [y] * n_rotations
            data_y = padding(data_y, max_length)

            data_x = np.vstack(data_x)
            data_x = np.reshape(
                data_x, (data_x.shape[0], data_x.shape[1], 1))

            result['x_'+subset] = data_x
            result['y_'+subset] = data_y

            yield result if result else None
