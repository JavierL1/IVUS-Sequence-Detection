from .dataset_managing import DataManager


def generate_signals(processed_data, models_path, model_name, save_dir):
    dataset_manager = DataManager(processed_data=processed_data)

    dataset_manager.predict_pullbacks_cnn(
        models_path, model_name, save_dir,
        train_rnn=True,
    )
