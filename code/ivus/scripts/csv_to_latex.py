import sys
import os
from pickle import load

from ivus.models.metrics import get_metrics

TOP_5_FILENAME = 'top_5_best.pkl'
HYPERPARAM_TABLE_FILENAME = '{}_hp.tex'
MODEL_SUFIX = 'hp_'

RESULTS_FOLDER = sys.argv[1]


def load_top_5_models(experiment_folder):
    top_5_path = os.path.join(experiment_folder, TOP_5_FILENAME)
    with open(top_5_path, 'rb') as top_5_pkl:
        return load(top_5_pkl)


def experiment_folders():
    for experiment_name in os.listdir(RESULTS_FOLDER):
        yield os.path.join(RESULTS_FOLDER, experiment_name)


class ExperimentLatexWriter:
    def __init__(self, experiment_folder):
        self.folder = experiment_folder
        self.name = self.folder.split('\\')[-1]
        self.keyword = self.name.replace(MODEL_SUFIX, '')
        self.top_5_models = load_top_5_models(self.folder)
        self.parameters_tex_path = os.path.join(
            self.folder, f'{self.name}_params.tex'
        )
        self.metrics_tex_path = os.path.join(
            self.folder, f'{self.name}_metrics.tex'
        )

    def _get_parameters_line(self, model_data):
        model_params = model_data['params']
        return (
            f'{self.keyword.upper()}-${model_data["model_id"]}$ & '
            f'${model_params["beta_1"]:.3e}$ & '
            f'${model_params["recurrent_dropout"]}$ & '
            f'${model_params["lr"]:.3e}$ & '
            f'${model_params["lstm_units"]}$ \\\\\n'
        )

    def _write_parameters_line(self, model_data):
        with open(self.parameters_tex_path, 'a') as tex_file:
            tex_file.write(self._get_parameters_line(model_data))

    def write_parameters_table(self):
        for model_data in self.top_5_models:
            self._write_parameters_line(model_data)

    def _get_metrics_values_line(self, model_metrics, subset):
        return (
            f'${model_metrics[subset]["f1_medians"]:.2f}\,\,({model_metrics[subset]["f1_medians_std"]:.2f})$ & '
            f'${model_metrics[subset]["pre_medians"]:.2f}\,\,({model_metrics[subset]["pre_medians_std"]:.2f})$ & '
            f'${model_metrics[subset]["rec_medians"]:.2f}\,\,({model_metrics[subset]["rec_medians_std"]:.2f})$'
        )

    def _get_metrics_line(self, model_id, model_metrics):
        return (
            f'{self.keyword.upper()}-${model_id}$'
            f' & {self._get_metrics_values_line(model_metrics, "val")}'
            f' & {metrics_line} & {self._get_metrics_values_line(model_metrics, "test")} \\\\\n'
        )

    def _write_metrics_line(self, model_id):
        model_metrics = get_metrics(self.name, model_id)
        with open(self.metrics_tex_path, 'a') as tex_file:
            tex_file.write(self._get_metrics_line(model_id, model_metrics))

    def write_metrics_table(self):
        model_ids = [model_data['model_id'] for model_data in self.top_5_models]
        for model_id in model_ids:
            self._write_metrics_line(model_id)


def write_latex_tables():
    for experiment_folder in experiment_folders():
        latex_writer = ExperimentLatexWriter(experiment_folder)
        latex_writer.write_parameters_table()
        latex_writer.write_metrics_table()
        return

if __name__ == "__main__":
    write_latex_tables()
