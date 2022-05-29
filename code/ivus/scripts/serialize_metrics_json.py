import json
import os

from ivus.models.model_metrics import get_metrics

RESULTS_FOLDER = os.getenv('RESULTS_FOLDER')

def get_json_path(experiment_name, model_id):
    return os.path.join(
        RESULTS_FOLDER,
        experiment_name,
        'model_{}'.format(model_id),
        'metrics.json'
    )

def get_experiment_names():
    return [
        dir_name
        for dir_name in os.listdir(RESULTS_FOLDER)
        if dir_name.startswith('hp_')
    ]

def get_model_ids(experiment_name):
    return [
        model_name.split('_')[-1]
        for model_name in os.listdir(os.path.join(RESULTS_FOLDER, experiment_name))
        if model_name.startswith('model_')
    ]

def serialize_metrics(metrics):
    return {
        fold: {
            subset: {
                pb_name: {
                    **{
                        field: float(metrics[fold][subset][pb_name][field])
                        for field in metrics[fold][subset][pb_name]
                        if field not in ['real', 'pred']
                    },
                    'real': metrics[fold][subset][pb_name]['real'].tolist(),
                    'pred': metrics[fold][subset][pb_name]['pred'].tolist(),
                } for pb_name in metrics[fold][subset].keys()
            } for subset in metrics[fold].keys()
        } for fold in metrics.keys()
    }

def serialize_subsets_metrics(metrics):
    return {
        subset: {
            metric_type: float(metrics[subset][metric_type])
            for metric_type in metrics[subset].keys()
        } for subset in metrics.keys()
    }

def get_serialized_object(model_metrics):
    return {
        'metrics': serialize_metrics(model_metrics.metrics),
        'subsets_metrics': serialize_subsets_metrics(model_metrics.subsets_metrics)
    }

def serialize_to_json(experiment_name, model_id):
    model_metrics = get_metrics(experiment_name, model_id)
    serialized_object = get_serialized_object(model_metrics)
    with open(get_json_path(experiment_name, model_id), 'w') as json_file:
        json.dump(serialized_object, json_file)

def serialize_all_to_json():
    for experiment_name in get_experiment_names():
        for model_id in get_model_ids(experiment_name):
            serialize_to_json(experiment_name, model_id)


if __name__ == "__main__":
    serialize_all_to_json()
