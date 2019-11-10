import os
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from keras.models import load_model

load_dotenv(find_dotenv())


def make_predictions(
    base_path, objects, data_generator, tf_length=None, **kwargs
):
    subsets = [subset for subset, value in kwargs.items() if value]
    predictions_path = os.path.join(base_path, 'predictions')
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    for dataset in tqdm(
        data_generator() if not tf_length else data_generator(tf_length)
    ):

        fold_model_path = os.path.join(
            base_path,
            'models',
            'fold_{}_best.model'.format(dataset['fold'])
        )
        model = load_model(fold_model_path, custom_objects=objects)

        for subset in subsets:

            y_pred = model.predict(dataset['x_'+subset])
            save_folder = os.path.join(
                predictions_path,
                '{}/{}'.format(
                    dataset['fold'], subset.upper())
            )
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for pb_index in range(dataset['y_'+subset].shape[0]):
                save_file = os.path.join(
                    save_folder, dataset['names_'+subset][pb_index]+'.csv')
                with open(save_file, 'w') as csvfile:
                    for frame in range(dataset['y_'+subset].shape[1]):
                        real_value = dataset['y_'+subset][pb_index][frame][0]
                        if real_value == -1:
                            break
                        csvfile.write(
                            '{}, {}\n'.format(
                                y_pred[pb_index][frame][0],
                                real_value
                            ))
