import os
from dotenv import load_dotenv, find_dotenv
from ivus.data.preprocessing import prepare_pullbacks

load_dotenv(find_dotenv())

RAW_DATA = os.getenv("RAW_DATA")
PROCESSED_DATA = os.getenv("PROCESSED_DATA")
LABELS_DATA = os.getenv("LABELS_DATA")


def preprocess():
    prepare_pullbacks(
        input_folder=RAW_DATA,
        filename_labels=os.path.join(LABELS_DATA, 'clean_labels_cougar.txt'),
        output_folder=PROCESSED_DATA,
        n_features=3,
        pixels_in=512,
        pixels_out=128
    )


if __name__ == '__main__':
    preprocess()
