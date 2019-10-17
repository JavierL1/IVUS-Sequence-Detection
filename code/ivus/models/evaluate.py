import os
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv, find_dotenv
from keras.models import load_model

load_dotenv(find_dotenv())


def get_metrics_subset(y_true, y_pred):
    y_true = np.reshape(y_true,
                        (y_true.shape[0], y_true.shape[1]))
    y_pred = np.reshape(y_pred,
                        (y_pred.shape[0], y_pred.shape[1]))

    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    _tp = []
    _fp = []
    _fn = []

    for (seq_true, seq_hat) in zip(y_true, y_pred):
        finder = np.where(seq_true == -1)[0]
        end = finder[0] if len(finder) > 0 else len(y_true)
        _tp.append(np.sum(seq_true[:end]*seq_hat[:end]))
        _fp.append(np.sum((1-seq_true[:end])*seq_hat[:end]))
        _fn.append(np.sum((1-seq_hat[:end])*seq_true[:end]))

    _pre = []
    _rec = []

    for (tp, fp, fn) in zip(_tp, _fp, _fn):
        _pre.append(1 if tp+fp == 0 else tp/(tp+fp))
        _rec.append(1 if tp+fn == 0 else tp/(tp+fn))

    _f1 = []

    for (pre, rec, tp, fp, fn) in zip(_pre, _rec, _tp, _fp, _fn):
        _f1.append(
            1 if (tp + fp + fn == 0) else (
                0 if (pre == 0 or rec == 0) else 2*pre*rec/(pre+rec)
            ))

    return (_pre, _rec, _f1)


def get_metrics(
    base_path, objects, data_generator, subsets, best=False
):

    model_name_end = '_best.model' if best else '.model'

    metrics = {
        subset: {
            version: {
                type_metric: []
                for type_metric in ['pre', 'rec', 'f1']
            } for version in ['old', 'new']
        } for subset in subsets
    }

    for dataset in tqdm(data_generator()):

        fold_model_path = os.path.join(
            base_path,
            'fold_{}{}'.format(dataset['fold'], model_name_end)
        )
        model = load_model(fold_model_path, custom_objects=objects)

        for subset in subsets:

            y_pred = model.predict(dataset['x_'+subset])

            (pre, rec, f1) = get_metrics_subset(dataset['y_'+subset], y_pred)

            metrics[subset]['old']['pre'].append(np.median(pre))
            metrics[subset]['old']['rec'].append(np.median(rec))
            metrics[subset]['old']['f1'].append(np.median(f1))

            metrics[subset]['new']['pre'] = np.concatenate(
                (metrics[subset]['new']['pre'], pre))
            metrics[subset]['new']['rec'] = np.concatenate(
                (metrics[subset]['new']['rec'], rec))
            metrics[subset]['new']['f1'] = np.concatenate(
                (metrics[subset]['new']['f1'], f1))

    for subset in subsets:
        for type_metric in ['pre', 'rec', 'f1']:
            for version in ['old', 'new']:
                metrics[subset][version][type_metric+'_std'] = np.std(
                    metrics[subset][version][type_metric]
                )
                metrics[subset][version][type_metric] = np.mean(
                    metrics[subset][version][type_metric]
                )

    return metrics
