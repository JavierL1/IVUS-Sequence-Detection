import csv
import numpy as np
import os


def read_data_subsets(
    filename, n_features=1, has_header=False,
    max_len_forced=0
):

    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        if has_header:
            next(reader)
        seq_lengths = dict()
        seq_indices = dict()
        for row in reader:
            name_elements = row[0]
            seq_name = str(name_elements)
            if seq_name in seq_lengths:
                seq_lengths[seq_name] += 1
            else:
                seq_lengths[seq_name] = 1
                seq_indices[seq_name] = len(seq_indices)

    n_sequences = len(seq_lengths)

    lens = list(seq_lengths.values())
    max_len = max(lens)

    if max_len_forced != 0:
        max_len = max_len_forced

    data = -np.ones((n_sequences, max_len, 2*n_features), dtype=np.float32)

    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        if has_header:
            next(reader)
        pointers_write = np.zeros(n_sequences, dtype=np.int32)

        for row in reader:
            name_elements = row[0]
            seq_name = str(name_elements)
            seq_idx = seq_indices[seq_name]
            pos_write = pointers_write[seq_idx]
            data[seq_idx, pos_write, 0:n_features] = np.array(
                [float(i) for i in row[1:n_features+1]])
            data[seq_idx, pos_write, n_features:] = np.array(
                [int(i) for i in row[n_features+1:2*n_features+1]])
            pointers_write[seq_idx] += 1

    return {
        'data': data,
        'n_sequences': n_sequences,
        'max_len': max_len,
    }


def read_data_pullbacks(folder, pad=1):
    files_list = os.listdir(folder)
    n_sequences = len(files_list)
    max_length = 0

    all_predicted = []
    all_real = []

    predicted = []
    real = []

    for _file in files_list:
        length = 0
        with open(folder+'/'+_file) as pullback_file:
            pullback_reader = csv.reader(pullback_file, delimiter=',')
            for row in pullback_reader:
                length += 1
                predicted.append(float(row[0]))
                real.append(float(row[1]))

        max_length = np.max([max_length, length])

        all_predicted.append(np.array(predicted, dtype=np.float32))
        all_real.append(np.array(real, dtype=np.float32))
        predicted = []
        real = []

    lengths = [len(seq) for seq in all_real]

    if pad == 0:
        return {
            'x': all_predicted,
            'y': all_real,
            'n_sequences': n_sequences,
            'max_len': max_length,
            'lengths': lengths,
            'names': [filename.split('.')[0] for filename in files_list],
        }

    elif pad == 1:
        all_predicted = [
            np.pad(
                sequence, (0, max_length-len(sequence)), 'constant',
                constant_values=-1)
            for sequence in all_predicted
        ]

        all_real = [
            np.pad(
                sequence, (0, max_length-len(sequence)), 'constant',
                constant_values=-1)
            for sequence in all_real
        ]

        x = np.stack(all_predicted)
        y = np.stack(all_real)

        return {
            'x': x,
            'y': y,
            'n_sequences': n_sequences,
            'max_len': max_length,
            'lengths': lengths,
            'names': [filename.split('.')[0] for filename in files_list],
        }

    elif pad == 2:
        all_predicted = [
            np.pad(
                sequence, (max_length-len(sequence), 0), 'constant',
                constant_values=-1)
            for sequence in all_predicted
        ]

        all_real = [
            np.pad(
                sequence, (max_length-len(sequence), 0), 'constant',
                constant_values=-1)
            for sequence in all_real
        ]

        x = np.stack(all_predicted)
        y = np.stack(all_real)

        return {
            'x': x,
            'y': y,
            'n_sequences': n_sequences,
            'max_len': max_length,
            'lengths': lengths,
            'names': [filename.split('.')[0] for filename in files_list],
        }



def sequence_fourier_padding(sequence, size):
    seq_len = len(sequence)
    if seq_len >= size:
        return sequence[-size:]
    else:
        repeats = size // seq_len
        remainder = size % seq_len
        output = []
        if remainder != 0:
            output.extend(sequence[-remainder:])
        for i in range(repeats):
            output.extend(sequence)
        return output


def time_folder_fourier(data, fold_length):
    output = []
    for sequence in data:
        seq_len = sequence.shape[0]
        padding = sequence_fourier_padding(sequence, fold_length)
        folded_seq = []
        for index in range(seq_len):
            if index < fold_length-1:
                seq_list = [
                    item for _list in [
                        padding[index+1:],
                        sequence[0:index+1]
                    ] for item in _list
                ]
                folded_seq.append(
                    np.array(seq_list, dtype=np.float32))
            else:
                folded_seq.append(sequence[index+1-fold_length:index+1])
        output.append(folded_seq)
    return output


def vector_padding(x_data, y_data, max_len, vector_len):
    padding = -np.ones((vector_len))
    x_sequences = x_data
    x_out = []
    for sequence in x_sequences:
        for pad in range(max_len - len(sequence)):
            sequence.append(padding)
        x_out.append(
            np.vstack(sequence)
        )
    x_out = np.stack(x_out)

    y_sequences = y_data
    y_out = [
        np.pad(sequence, (0, max_len - sequence.shape[0]),
               'constant', constant_values=-1)
        for sequence in y_sequences
    ]
    y_out = np.reshape(np.stack(y_out), (len(y_out), max_len, 1))

    return x_out, y_out


def shifts(x, y):
    x_shift = []
    y_shift = []
    max_len = 0
    for x_seq, y_seq in zip(x, y):
        curr_len = len(x_seq)
        if max_len < curr_len:
            max_len = curr_len
        for shift_len in range(curr_len):
            x_shift.append(np.roll(x_seq, shift_len))
            y_shift.append(np.roll(y_seq, shift_len))

    return x_shift, y_shift, max_len


def padding(sequences, max_len):
    padded = [
        np.pad(seq, (0, max_len - len(seq)),
               'constant', constant_values=-1)
        for seq in sequences
    ]

    return np.reshape(np.stack(padded), (len(padded), max_len, 1))


def cropping(sequences, window_size, stride=1):
    cropped = []
    n_shorts = 0
    n_longs = 0
    for seq in sequences:
        seq_len = len(seq)
        if window_size > seq_len:
            n_shorts += 1
            cropped.append(np.pad(
                seq, (0, window_size - seq_len), 'constant',
                constant_values=-1
            ))
        else:
            n_longs += 1
            n_windows = (seq_len - window_size) / stride + 1
            for window in [
                seq[0+index:window_size+index]
                for index in range(n_windows)
            ]:
                cropped.append(window)
    return (
        np.reshape(np.stack(cropped), (len(cropped), window_size, 1)),
        {
            'shorts': n_shorts,
            'longs': n_longs
        }
    )
