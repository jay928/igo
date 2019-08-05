from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Reshape, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Embedding
from keras.utils import np_utils
from numpy import array
import numpy as np
import glob
import os
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # for Mac

VOCA_PATH = "/Users/JangwonPark/Documents/tensorflow/allvocaset.csv"
DATA_PATH = "/Users/JangwonPark/Documents/tensorflow/data_sample_small/raw"
DETAIL_PATH = '/blog.naver.com/*.txt'

embedding_size, cnn_kerner_size = 32, (10, 5)
verbose, epochs, batch_size = 0, 20, 32


def npIndex(npArray, object):
    result = np.where(npArray==object)
    return np.array(result).tolist()[0][0] + 1


def findFolders(path):
    folders = []
    for r, d, f in os.walk(path):
        for folder in d:
            folderName = os.path.join(r, folder)
            if "blog.naver.com" in folderName:
                continue

            folders.append(folderName)

    return folders


def findLabels(folders):
    labels = []
    for folder in folders:
        label = folder.split("/")[-1].split(" ")[0]
        labels.append(label)

    return array(list(set(labels)))


def findY(y_array, file):
    for y in y_array:
        if y in file:
            return y

    return None


def loadData(folders, labels, voca):
    x_data = []
    y_data = []

    for i, folder in enumerate(folders):
        files = [f for f in glob.glob(folder + DETAIL_PATH, recursive=True)]

        for j, file in enumerate(files):
            print("(", j+1, "/", len(files), ", ", i+1, "/", len(folders), ") Importing file : ", file)

            y = findY(labels, file)
            y_as_int = npIndex(labels, y)

            with open(file, 'r', encoding='UTF-8') as f:
                lines = f.readlines()
                for line in lines:
                    x_line = []

                    for index in range(0, len(line)):
                        char = line[index]
                        if char not in voca:
                            continue

                        x_line.append(npIndex(voca, char))

                    if not x_line:
                        continue

                    x_data.append(x_line)
                    y_data.append(y_as_int)

    return x_data, y_data


voca_arr = np.loadtxt(VOCA_PATH, dtype=np.str, encoding='UTF-8')
folder_names = findFolders(DATA_PATH)
label_names = findLabels(folder_names)

x_data, y_data = loadData(folder_names, label_names, voca_arr)

x_data = sequence.pad_sequences(x_data, dtype='int32', padding='post', truncating='post', value=0)
y_data = np_utils.to_categorical(y_data)

split_idx = int(len(x_data) * 0.7)

x_train, y_train = x_data[:split_idx], y_data[:split_idx]
x_validation, y_validation = x_data[split_idx:], y_data[split_idx:]

voca_length, input_length, label_length = len(voca_arr) + 1, len(x_data[0]), len(label_names)


model = Sequential()
model.add(Embedding(voca_length, embedding_size, input_length=input_length))
model.add(Reshape((input_length, embedding_size, 1)))
model.add(Conv2D(filters=2*32, kernel_size=cnn_kerner_size, activation='relu', input_shape=x_train.shape, padding='valid', strides=1))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=3*32, kernel_size=cnn_kerner_size, activation='relu', padding='valid', strides=1))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(label_length, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)

print("Tranining Completed!")
print('\nAccuracy: {:.4f}'.format(model.evaluate(x_validation, y_validation)[1]))
