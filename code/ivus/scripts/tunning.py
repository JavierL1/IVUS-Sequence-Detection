from dataclasses import dataclass, field, InitVar
import csv
import cPickle
import json
import os
import sys

from dotenv import load_dotenv, find_dotenv
from scipy.ndimage.filters import gaussian_filter1d
from tqdm import tqdm
import numpy as np

from ivus.models.model_metrics import compute_confusion_matrix, compute_praf

OUTPUT_FILENAME = sys.argv[1]
TUNNING_SUBSET = sys.argv[2] or 'train_cnn'
load_dotenv(find_dotenv())
CNN_PREDICTIONS = os.getenv('CNN_PREDICTIONS')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER')
SUBSETS = ['train_cnn', 'val', 'train_rnn', 'test']
FOLDS = range(1, 11)
METRIC_TYPES = ['acc', 'f1', 'pre', 'rec']

theta_min = 0.1
theta_max = 0.9
sigma_min = 0.1
sigma_max = 8
grid_density = 100

theta_space = np.linspace(theta_min, theta_max, grid_density)
sigma_space = np.linspace(sigma_min, sigma_max, grid_density)
number_experiments = len(theta_space) * len(sigma_space)


@dataclass
class Pullback:
    name: str
    cnn_predictions: np.ndarray
    groundtruth: np.ndarray
    pre: np.float32 = field(init=False)
    rec: np.float32 = field(init=False)
    acc: np.float32 = field(init=False)
    f1: np.float32 = field(init=False)

    def __post_init__(self, theta, sigma):
        cnn_smoothed = gaussian_filter1d(
            self.cnn_predictions,
            sigma=sigma,
            mode='constant',
        )
        praf = compute_praf(compute_confusion_matrix(
            self.groundtruth,
            cnn_smoothed,
            threshold=theta,
        ))
        self.pre = praf['pre']
        self.rec = praf['rec']
        self.acc = praf['acc']
        self.pre = praf['f1']


@dataclass
class MetricsStats:
    mean: np.float32 = field(init=False)
    median: np.float32 = field(init=False)
    q1: np.float32 = field(init=False)
    q3: np.float32 = field(init=False)
    metric_type: InitVar[str] = None
    metric_values: InitVar[list(Pullback)] = None

    def __post_init__(self, metric_type, metric_values):
        self.mean = np.mean(metric_values)
        self.median = np.median(metric_values)
        self.q1 = np.percentile(metric_values, 25)
        self.q3 = np.percentile(metric_values, 75)


@dataclass
class FoldMetrics:
    fold: int
    theta: np.float32
    sigma: np.float32
    pre: MetricsStats = field(init=False)
    rec: MetricsStats = field(init=False)
    acc: MetricsStats = field(init=False)
    f1: MetricsStats = field(init=False)
    fold_pullbacks: InitVar[list(Pullback)] = None

    def __post_init__(self, fold_pullbacks):
        for metric_type in METRIC_TYPES:
            self.__setattr__(
                metric_type,
                MetricsStats(
                    metric_type,
                    [
                        pullback.getattr(metric_type)
                        for pullback in fold_pullbacks
                    ],
                )
            )


def read_pullback(csv_reader, pullback_name):
    cnn_predictions = []
    groundtruth = []
    for row in csv_reader:
        cnn_predictions.append(row['pred'])
        groundtruth.append(row['real'])

    return Pullback(
        name=pullback_name,
        cnn_predictions=np.array(cnn_predictions),
        groundtruth=np.array(groundtruth)
    )


def iter_pullbacks(fold, subset):
    reader_config = {
        'fieldnames': ['pred', 'real'],
        'delimiter': ',',
    }
    folder_path = os.path.join(CNN_PREDICTIONS, str(fold), 'CSV', subset.upper())
    for filename in os.listdir(folder_path):
        pullback_name = filename.split('.')[0]
        with open(os.path.join(folder_path, filename)) as csv_file:
            csv_reader = csv.DictReader(csv_file, **reader_config)
            yield read_pullback(csv_reader, pullback_name)


def tune_fold(fold, subset):
    for theta in tqdm(theta_space):
        for sigma in tqdm(sigma_space):
            fold_pullbacks = [
                pullback
                for pullback in iter_pullbacks(fold, subset):
            ]
            return FoldMetrics(
                fold,
                theta,
                sigma,
                fold_pullbacks=fold_pullbacks
            )


def full_tunning():
    return [tune_fold(fold, TUNNING_SUBSET) for fold in FOLDS]


if __name__ == "__main__":
    tunning = full_tunning()
    output_path = os.path.join(RESULTS_FOLDER, f'{OUTPUT_FILENAME}.pkl')
    with open(output_path, 'wb') as output_file:
        cPickle.dump(tunning, output_file)
