import os
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from keras.models import load_model

load_dotenv(find_dotenv())


def make_predictions(
    base_path, model_id, objects, data_generator, subsets, best=True
):

    model_name_end = '_best.model' if best else '.model'

    predictions_path = os.path.join(base_path, 'Predictions')
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    for dataset in tqdm(data_generator()):

        fold_model_path = os.path.join(
            base_path,
            'models',
            'fold_{}{}'.format(dataset['fold'], model_name_end)
        )
        model = load_model(fold_model_path, custom_objects=objects)

        for subset in subsets:

            y_pred = model.predict(dataset['x_'+subset])
            save_folder = os.path.join(
                predictions_path,
                '{}/{}/CSV/{}'.format(
                    model_id, dataset['fold'], subset.upper())
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
