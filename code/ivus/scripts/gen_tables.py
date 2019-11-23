import os
from dotenv import load_dotenv, find_dotenv
from ivus.models.metrics import get_metrics

load_dotenv(find_dotenv())
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER')
SUBSETS = ['train_cnn', 'val', 'train_rnn', 'test']
HEADERS = [
    '{0}{1}{2}'.format(measure, _type, end)
    for measure in ['f1', 'pre', 'rec', 'acc']
    for _type in ['_means', '_medians']
    for end in ['', '_std']
]

experiment_names = [
    filename
    for filename in os.listdir(RESULTS_FOLDER)
    if filename.startswith('hp')
]

for exp_name in experiment_names:
    exp_folder = os.path.join(RESULTS_FOLDER, exp_name)
    model_ids = [
        filename.split('_')[-1]
        for filename in os.listdir(exp_folder)
        if filename.startswith('model')
    ]
    model_metrics = {
        model_id: get_metrics(exp_name, model_id)
        for model_id in model_ids
    }
    for subset in SUBSETS:
        write_headers = True
        with open(os.path.join(
            exp_folder,
            '{}_{}_table.csv'.format(exp_name.split('_', 1)[-1], subset)),
            'w'
        ) as csv_table:
            for model_id in model_ids:
                subset_metrics = model_metrics[
                    model_id].subsets_metrics[subset]
                if write_headers:
                    csv_table.write('model_id')
                    for header in HEADERS:
                        csv_table.write(', {}'.format(header))
                    csv_table.write('\n')
                    write_headers = False
                csv_table.write(model_id)
                for header in HEADERS:
                    csv_table.write(', {0:.2}'.format(subset_metrics[header]))
                csv_table.write('\n')
