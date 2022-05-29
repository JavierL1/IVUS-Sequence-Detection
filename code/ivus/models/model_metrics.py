import csv
import numpy as np
import os
import pickle
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER')
SUBSETS = ['train_cnn', 'val', 'train_rnn', 'test']
FOLDS = range(1, 11)
BEST_MODELS = [
    ('hyperopt_rnn', '465'),
    ('hyperopt_rnn_pre', '251'),
    ('hyperopt_tf_rnn', '319'),
    ('mode_best_models', None)
]

def compute_confusion_matrix(y_true, y_pred, threshold=0.5):
    classes = ['tp', 'fn', 'fp', 'tn']
    conf_matrix = {
        _class: np.float32(0)
        for _class in classes
    }
    for y_t, y_p in zip(y_true, y_pred):
        if y_t >= threshold:
            if y_p >= threshold:
                conf_matrix['tp'] += 1
            else:
                conf_matrix['fn'] += 1
        else:
            if y_p >= threshold:
                conf_matrix['fp'] += 1
            else:
                conf_matrix['tn'] += 1

    conf_matrix['total'] = np.float32(len(y_true))

    return conf_matrix


def compute_praf(conf_matrix):
    types = ['pre', 'rec', 'acc', 'f1']
    metrics = {
        _type: np.float32(0)
        for _type in types
    }

    if conf_matrix['tp'] + conf_matrix['fp'] > 0:
        metrics['pre'] = (
            conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fp']))
    else:
        metrics['pre'] = 1.0

    if conf_matrix['tp'] + conf_matrix['fn'] > 0:
        metrics['rec'] = (
            conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn']))
    else:
        metrics['rec'] = 1.0

    metrics['acc'] = (
        (conf_matrix['tp'] + conf_matrix['tn']) / conf_matrix['total'])

    if metrics['pre'] + metrics['rec'] > 0:
        metrics['f1'] = (2 * (
                (metrics['pre'] * metrics['rec'])
                / (metrics['pre'] + metrics['rec'])))
    else:
        metrics['f1'] = 0.0

    return metrics


def certainty(prob):
    return np.abs(prob - 0.5)/0.5


def calculate_pb_metrics(pbs_path, pb_filename):
    pb_path = os.path.join(pbs_path, pb_filename)
    pred = []
    real = []
    with open(pb_path, 'r') as pb_file:
        csv_lines = csv.reader(pb_file, delimiter=',')
        for line in csv_lines:
            pred.append(np.float32(line[0]))
            real.append(np.float32(line[1]))
    pred = np.array(pred)
    real = np.array(real)
    conf_matrix = compute_confusion_matrix(real, pred)
    praf = compute_praf(conf_matrix)
    conf_matrix['pred'] = pred
    conf_matrix['real'] = real
    return dict(conf_matrix, **praf)


def jaccard_score(true, pred):
    true = np.floor(true + 0.5)
    pred = np.floor(pred + 0.5)
    inter = sum(true * pred)
    union = sum(np.ceil((true + pred) / 2.0))
    if union == 0.0:
        return 1.0
    else:
        return inter/union


class ModelMetrics(object):
    def __init__(self, exp_name, model_id=None):
        self.exp_name = exp_name
        if model_id is None:
            self.model_folder = os.path.join(
                RESULTS_FOLDER,
                exp_name
            )
            self.preds_path = os.path.join(
                self.model_folder,
                'predictions'
            )
        else:
            self.model_id = model_id
            self.model_folder = os.path.join(
                RESULTS_FOLDER,
                exp_name,
                'model_{}'.format(model_id)
            )
            self.preds_path = os.path.join(
                self.model_folder,
                'predictions'
            )
        self.calculate_all_pullback_metrics()
        self.calculate_subsets_metrics()

    def save(self):
        with open(
            os.path.join(self.model_folder, 'metrics.pkl'),
            'wb'
        ) as metrics_file:
            pickle.dump(self, metrics_file)

    def calculate_all_pullback_metrics(self, threshold=0.5):
        self.metrics = {
            fold: {
                subset: {}
                for subset in SUBSETS
            } for fold in FOLDS
        }
        for fold in FOLDS:
            for subset in SUBSETS:
                pbs_path = os.path.join(
                    self.preds_path,
                    str(fold),
                    subset.upper()
                )
                for pb_filename in os.listdir(pbs_path):
                    pb_name = pb_filename.split('.')[0]
                    self.metrics[fold][subset][pb_name] = calculate_pb_metrics(
                            pbs_path,
                            pb_filename
                        )

    def calculate_subsets_metrics(self):
        types = ['pre', 'rec', 'acc', 'f1']
        self.subsets_metrics = {}
        for subset in SUBSETS:
            self.subsets_metrics[subset] = {}
            for _type in types:
                mean_values = [
                    np.mean([
                        self.metrics[fold][subset][pb_name][_type]
                        for pb_name in self.metrics[fold][subset].keys()
                    ])
                    for fold in FOLDS
                ]
                self.subsets_metrics[subset].update({
                    '{}_{}'.format(_type, meas_name): meas_fun(mean_values)
                    for meas_name, meas_fun in zip(
                        ['means', 'means_std'],
                        [np.mean, np.std])
                })
                median_values = [
                    np.median([
                        self.metrics[fold][subset][pb_name][_type]
                        for pb_name in self.metrics[fold][subset].keys()
                    ])
                    for fold in FOLDS
                ]
                self.subsets_metrics[subset].update({
                    '{}_{}'.format(_type, meas_name): meas_fun(median_values)
                    for meas_name, meas_fun in zip(
                        ['medians', 'medians_std'],
                        [np.mean, np.std])
                })

    def calculate_jaccard(self, exp_name, model_id=None):
        competitor = get_metrics(exp_name, model_id=model_id)
        jaccard_scores = {
            subset: {
                'values': []
            }
            for subset in SUBSETS
        }

        for fold in FOLDS:
            for subset in SUBSETS:
                fold_subset = self.metrics[fold][subset]
                for pb_name in fold_subset.keys():
                    jaccard_scores[subset]['values'].append(jaccard_score(
                        fold_subset[pb_name]['pred'],
                        competitor.metrics[fold][subset][pb_name]['pred']
                    ))

        for subset in SUBSETS:
            jaccard_scores[subset]['values'] = np.array(
                jaccard_scores[subset]['values'])
            jaccard_scores[subset]['mean'] = np.mean(
                jaccard_scores[subset]['values'])
            jaccard_scores[subset]['std'] = np.std(
                jaccard_scores[subset]['values'])

        return jaccard_scores


def get_metrics(exp_name, model_id=None):
    if model_id is None:
        with open(os.path.join(
                RESULTS_FOLDER,
                exp_name,
                'metrics.pkl'
        ), 'rb') as metrics_file:
            metrics = pickle.load(metrics_file)
        return metrics
    else:
        with open(os.path.join(
                RESULTS_FOLDER,
                exp_name,
                'model_{}'.format(model_id),
                'metrics.pkl'
        ), 'rb') as metrics_file:
            metrics = pickle.load(metrics_file)
        return metrics


def calculate_jaccard_matrix():
    results = {}
    matrix_models = BEST_MODELS
    for exp_name_1, model_id_1 in matrix_models:
        results[exp_name_1] = {}
        model_metrics = get_metrics(exp_name_1, model_id=model_id_1)
        for exp_name_2, model_id_2 in matrix_models:
            results[exp_name_1][exp_name_2] = model_metrics.calculate_jaccard(
                exp_name_2, model_id=model_id_2)

    matrices = {
        subset: []
        for subset in SUBSETS
    }
    headers = []
    for exp_name_1, model_id_1 in matrix_models:
        headers.append(exp_name_1)
        row = {
            subset: []
            for subset in SUBSETS
        }
        for exp_name_2, model_id_2 in matrix_models:
            for subset in SUBSETS:
                if exp_name_1 != exp_name_2:
                    row[subset].append(
                        results[exp_name_1][exp_name_2][subset]['mean'])
                else:
                    row[subset].append(0.0)
        for subset in SUBSETS:
            matrices[subset].append(np.array(row[subset]))

    for subset in SUBSETS:
        matrices[subset] = np.stack(matrices[subset])

    return matrices, headers, results
