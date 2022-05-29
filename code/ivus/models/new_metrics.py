import json
import numpy as np
from os.path import join
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER')
SUBSETS = ['train_cnn', 'val', 'train_rnn', 'test']
FOLDS = range(1, 11)
BEST_MODELS = [
    ('hp_cr', '56'),
    ('hp_po', '63'),
    ('hp_pr', '29'),
    ('hp_ro', '14'),
    ('hp_sh', '66'),
    ('hp_tf', '5'),
]

def get_metrics(exp_name, model_id):
    metrics_path = join(
        RESULTS_FOLDER, exp_name, f'model_{model_id}', 'metrics.json'
    )
    with open(metrics_path, 'r') as metrics_json:
        metrics = json.load(metrics_json)

    return metrics

def jaccard_score(true, pred):
    true = np.floor(true + 0.5)
    pred = np.floor(pred + 0.5)
    inter = sum(true * pred)
    union = sum(np.ceil((true + pred) / 2.0))
    if union == 0.0:
        return 1.0
    else:
        return inter/union

def calculate_jaccard(exp_name_1, model_id_1, exp_name_2, model_id_2):
    competitor_1 = get_metrics(exp_name_1, model_id_1)
    competitor_2 = get_metrics(exp_name_2, model_id_2)
    jaccard_scores = {
        subset: {
            'values': []
        }
        for subset in SUBSETS
    }

    for fold in FOLDS:
        for subset in SUBSETS:
            fold_subset = competitor_1[fold][subset]
            for pb_name in fold_subset.keys():
                jaccard_scores[subset]['values'].append(jaccard_score(
                    fold_subset[pb_name]['pred'],
                    competitor_2[fold][subset][pb_name]['pred']
                ))

    for subset in SUBSETS:
        jaccard_scores[subset]['values'] = np.array(
            jaccard_scores[subset]['values'])
        jaccard_scores[subset]['mean'] = np.mean(
            jaccard_scores[subset]['values'])
        jaccard_scores[subset]['std'] = np.std(
            jaccard_scores[subset]['values'])

    return jaccard_scores

def calculate_jaccard_matrix():
    results = {}
    matrix_models = BEST_MODELS
    for exp_name_1, model_id_1 in matrix_models:
        results[exp_name_1] = {}
        for exp_name_2, model_id_2 in matrix_models:
            results[exp_name_1][exp_name_2] = calculate_jaccard(
                exp_name_1, model_id_1, exp_name_2, model_id_2
            )

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
