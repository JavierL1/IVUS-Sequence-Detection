import os
from dotenv import load_dotenv, find_dotenv
from ivus.models.metrics import ModelMetrics

load_dotenv(find_dotenv())
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER')

experiment_names = [
    filename
    for filename in os.listdir(RESULTS_FOLDER)
    if filename.startswith('hp')
]


for exp_name in experiment_names:
    model_ids = [
        filename.split('_')[-1]
        for filename in os.listdir(os.path.join(
            RESULTS_FOLDER,
            exp_name
        )) if filename.startswith('model')
    ]
    for model_id in model_ids:
        print(
            'Calculating metrics for {}\'s model {}'.format(
                exp_name,
                model_id
            ))
        model_metrics = ModelMetrics(exp_name, model_id)
        model_metrics.save()
        print('Metrics calculated')
