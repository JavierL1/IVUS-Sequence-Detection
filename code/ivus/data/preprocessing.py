from keras.preprocessing import image
from skimage.transform import resize
import csv
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np

load_dotenv(find_dotenv())

name_x = 'PULLBACKS_X_COUGAR.csv'
name_y = 'PULLBACKS_Y_COUGAR.csv'
name_names = 'PULLBACKS_NAMES_COUGAR.csv'
name_file_labels = '..\\labels\\clean_labels_cougar.txt'
name_file_labels_folder = '..'
output_folder = '.'
LABELS_DATA = os.getenv('LABELS_DATA')


# read folder with images, eventually reduce resolution and save results to CSV
# in addition it saves labels and names of pullbacks and frames in the same
# order as images have been written
def prepare_pullbacks(
    input_folder,
    filename_labels,
    output_folder,
    n_features,
    pixels_in,
    pixels_out
):

    # READ FRAMES
    # Get list of folders in directory
    subdirs = os.listdir('{}/DICOM_FRA'.format(input_folder))
    # Sort list of files
    subdirs.sort()

    # Create list of paths to frames in the form:
    #   input_folder/$n_$namepb/$namepb_$m.png
    # n: amount of pullbacks
    # m: amount of frames in pullback (may vary from pullback to pullback)
    count_frames = 0
    files = []

    print("Setting list of paths...")
    # For every folder(pullback)
    for dir_ in subdirs:
        # Build path to folder(pullback)
        subdir = os.path.join('{}/DICOM_FRA'.format(input_folder), dir_)
        # If folder(pullback) exists
        if os.path.isdir(subdir):
            # Get list of files(frames) inside pullback
            files_ = [
                os.path.join(subdir, f)
                for f in os.listdir(subdir)
                if os.path.isfile(os.path.join(subdir, f))
            ]
            # Sort list of files(frames)
            files_.sort()
            # For every file(frame) append to list of files
            for f in files_:
                files.append(f)
                count_frames += 1

    count_labels = 0

    # READ LABELS
    # Reading ground-truth
    # File in the format:
    #   FileName, Stent, Fibro, Calci, Bifurc, Healthy
    #   1 if feature exists or 0 if it doesn't
    print("Reading labels...")
    with open(filename_labels, 'r') as csvfile:
        # Read a file as a .csv with , as a delimiter
        reader = csv.reader(csvfile, delimiter=',')
        # Ignore first line
        next(reader)
        # For every line of file
        frame_labels = dict()
        for row in reader:
            # Count lines
            count_labels += 1
            # Paths in ubuntu already have '/' instead of '\\'
            frame_full_name = row[0]
            frame_full_name = "{}/{}".format(input_folder, frame_full_name)
            # Array with value of labels for current frame
            labels = np.array([int(i) for i in row[1:n_features+1]])
            # Populate dictionary in the form
            # $namepb_$m.png : [ 1/0 1/0 1/0 ]
            frame_labels[frame_full_name] = labels

    K = 1
    count_found = 0
    for f in files:
        if f in frame_labels:
            count_found += 1

    # Check whether labels and frames match
    print("LABEL FILE SIZE:%d" % count_labels)
    print("LABELS FOUND:%d" % count_found)
    print("FRAMES FOUND:%d" % count_frames)

    # Prepare input matrix
    #   Every row corresponds to one frame
    #   Every column to one pixel
    #   Usual size is 8914 x 16384
    X = np.zeros((count_found, pixels_out*pixels_out*K), dtype=np.float32)
    # Prepare output labels matrix
    #   Every row corresponds to one frame
    #   Every column corresponds to one feature
    #   Usual size is 8914 x 3
    Y = np.zeros((count_found, n_features), dtype=np.int32)

    n_rows = 0
    it = 0
    it_lim = 1000000000

    # Open blank csv file
    names_file = open('{}/{}'.format(output_folder, name_names), 'w')

    # For every fullpath to frame
    for f in files:
        # If frame has labels
        if f in frame_labels:
            # Generates an absolute path to frame
            img_path = os.path.abspath(f)
            # Loads frame in grayscale
            # target_size=(224, 224)
            img = image.load_img(img_path, grayscale=True)
            # Turns it into an array
            # Usualy of size 512x512x1
            x = image.img_to_array(img)
            # pixels_in corresponds to the expected input size of a square
            #   image
            # pixels_out corresponds to the expected size of the output square
            #   image
            # If their values is not equal, some processing is necessary
            if pixels_in != pixels_out:
                # Normalice pixel values
                x = x/255.0
                # Reshape from 512x512x1 to 512x512
                new_im = np.reshape(x, (pixels_in, pixels_in))
                # Create new frame resizing previous image to 128x128 using
                #   keras
                new_im_small = resize(
                    new_im,
                    (pixels_out, pixels_out),
                    order=1,
                    preserve_range=True
                )
                # Turn given image into a vector of length 128x128=16384
                new_im_small = np.reshape(new_im_small, pixels_out*pixels_out)
                # Denormalice pixel values ????
                x = 255.0*new_im_small
            else:
                # If shapes are equal resize from 512x512x1 to 512x512
                x = np.reshape(x, pixels_out*pixels_out)

            # Write vector of pixels onto $n_rows row of input matrix
            X[n_rows, :] = x
            # Write vector of labels onto $n_rows row of output matrix
            Y[n_rows, :] = frame_labels[f]

            # Get pullback and frame names from fullpath
            # elements_name = f.split('/',1)
            elements_name = f.split('/')
            # elements_name = elements_name[1].split('/',1)
            # pullback_name = elements_name[0]
            pullback_name = elements_name[-2]
            # frame_name = elements_name[1]
            frame_name = elements_name[-1]
            # Write csv file with every entry in the form:
            #   $n_$namepb, $namepb_$m.png
            names_file.write("%s,%s\n" % (pullback_name, frame_name))
            n_rows += 1

        it += 1

        if it > it_lim:
            break

    names_file.close()

    # Write text file with input matrix
    print("Writing X file...")
    np.savetxt(os.path.join(output_folder, name_x), X)
    # Write text file with output labels matrix
    print("Writing Y file...")
    np.savetxt(os.path.join(output_folder, name_y), Y, fmt='%d')
    print(X.shape)
    print(Y.shape)
    print("LABELS WRITTEN:%d" % n_rows)


class CSVReader:

    def __init__(self):
        print("creating CSVReader ... ")
        print("... Reader created!")

    # Reads pullbacks from a csv and creates an index that helps recognize the
    # pullbacks in the sequence of frames
    def read_pullbacks_from_CSV(
        self,
        names='PULLBACKS_NAMES.csv',
        file_x='PULLBACKS_X.csv',
        file_y='PULLBACKS_Y.csv',
        n_features=3,
        dim=128,
        return_names=False
    ):

        X, Y = self.read_fromCSV(
            csv_x=file_x, csv_y=file_y, dim=dim, n_features=n_features)

        start_indices = []
        end_indices = []

        pullback_names = []

        # Save all pullback names
        with open(names, 'r') as names_file:
            for names in names_file:
                pullback_name = names.split(",")[0]
                # Unused line
                # frame_name = names.split(",")[1]
                pullback_names.append(pullback_name)

        # As a pullback has many frames, this block
        # of code iterates over pullback names looking
        # for a change in them, which marks the end of
        # one pullback and the start of another one
        # The result is two arrays such that for a pullback i
        # start_indices[i]: marks begining of pullback i
        # end_indices[i]: marks ending of pullback i
        last_name = pullback_names[0]
        start_indices.append(0)
        for idx in range(1, len(pullback_names)):
            current_name = pullback_names[idx]
            if current_name != last_name:
                start_indices.append(int(idx))
                last_name = current_name
                end_indices.append(int(idx-1))

        end_indices.append(int(len(pullback_names)-1))

        if return_names:
            return X, Y, start_indices, end_indices, pullback_names

        return X, Y, start_indices, end_indices

    # Reconstructs images from rows of csv and populates multidim-matrix
    # with them.
    # Populates matrix of labels too.
    def read_fromCSV(
        self,
        csv_x='X_DICOM_FRA.csv',
        csv_y='Y_DICOM_FRA.csv',
        dim=128,
        n_features=3
    ):

        n_data = 0
        with open(csv_x, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                n_data += 1

        X = np.zeros((n_data, dim, dim, 1), dtype=np.float32)
        Y = np.zeros((n_data, n_features), dtype=np.float32)

        counter = 0
        with open(csv_x, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                pattern = np.reshape(
                    np.array(row, dtype=np.float32), (dim, dim))
                X[counter, :, :, 0] = pattern
                counter += 1

        counter = 0
        with open(csv_y, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                Y[counter, :] = np.array([float(i) for i in row])
                counter += 1

        print("In read_fromCSV")
        print(X.shape)
        print(Y.shape)

        return X, Y


def get_labels(observer):
    with open(os.path.join(
        LABELS_DATA, 'cougar_observer{}.txt'.format(observer)
    ), 'r') as labels_file:
        data = {}
        current_pb_name = ''
        for index, line in enumerate(labels_file):
            if index == 0:
                continue
            split_line = line.split(',')
            fr_path = split_line[0]
            fr_label = split_line[3]
            pb_name = fr_path.split('/')[1]
            if current_pb_name == '':
                current_pb_name = pb_name
                data[current_pb_name] = [float(fr_label)]
            elif pb_name != current_pb_name:
                data[current_pb_name] = np.array(data[current_pb_name])
                print(data[current_pb_name])
                current_pb_name = pb_name
                data[current_pb_name] = [float(fr_label)]
            else:
                data[current_pb_name].append(float(fr_label))
        data[current_pb_name] = np.array(data[current_pb_name])
        return data


if __name__ == '__main__':
    prepare_pullbacks(
        input_folder=name_file_labels_folder,
        filename_labels=name_file_labels,
        output_folder=output_folder,
        n_features=3,
        pixels_in=512,
        pixels_out=128
    )
